"""Fresh CPU processes make peak RSS comparable without retaining inputs or token tensors."""

import asyncio
import multiprocessing
import secrets
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path

from adaptive_llm.contracts import BenchmarkMeasurement, ModelManifest
from adaptive_llm.distillation.benchmark import measure
from adaptive_llm.evaluation.runner import Case, DiscardEvents, PipelineRunner
from adaptive_llm.evaluation.service import EvaluationDeployment
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.policy.persistence import LocalPersistenceRedactor
from adaptive_llm.research.service import BaseGenerator, SourceProvider
from adaptive_llm.routing import Deployment, FoundationRouter
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore
from adaptive_llm.training.lora import BaseFiles, LoraGenerator, cpu
from adaptive_llm.validation import Validator


@dataclass(frozen=True)
class MeasurementInput:
    model: ModelManifest
    files: dict[str, bytes] = field(repr=False)
    raw_base: BaseFiles | None = field(repr=False)
    deployment: Deployment
    cases: list[Case] = field(repr=False)
    identity: Identity
    validator: Validator
    routing_path: Path
    data_dir: Path
    requests: int


def worker(pipe: Connection, work: MeasurementInput) -> None:
    """All input stays in process memory; only aggregate measurements return over the pipe."""
    database = SQLiteDatabase(Path("."), in_memory=True)
    try:
        generator = (
            BaseGenerator(work.model, work.raw_base)
            if work.raw_base is not None
            else LoraGenerator(work.model, work.data_dir, work.files)
        )
        if work.model.adapter_architecture == "lora-peft-v1":
            with cpu(generator.libs, work.model.seed):
                generator.model = generator.model.merge_and_unload()
        provider = SourceProvider(generator)
        discard = DiscardEvents()
        persistence = Persistence(
            SQLiteMetadataStore(database),
            SQLitePayloadStore(database),
            PayloadCipher(secrets.token_bytes(32)),
            LocalPersistenceRedactor(),
            Keyring(secrets.token_bytes(32)),
            outbox=discard,
            fallback_sink=discard,
            metrics=InProcessMetrics(),
        )
        router = FoundationRouter(work.routing_path)
        router.deployment = work.deployment
        runner = PipelineRunner(
            persistence, work.identity, provider, router, work.validator, work.deployment.price_list
        )

        async def run() -> tuple[BenchmarkMeasurement, list[float]]:
            await runner.run(work.cases[0])
            return await measure(
                runner, EvaluationDeployment(work.deployment, provider), work.cases, work.requests
            )

        pipe.send(asyncio.run(run()))
    except Exception:
        pipe.send("research_measurement_failed")
    finally:
        database.close()
        pipe.close()


def isolated_measure(
    work: MeasurementInput, timeout: float
) -> tuple[BenchmarkMeasurement, list[float]]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender, work), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise GatewayError(503, "research_measurement_timeout")
        result = receiver.recv()
        if not isinstance(result, tuple) or len(result) != 2:
            raise GatewayError(503, "research_measurement_failed")
        measurement, scores = result
        if not isinstance(measurement, BenchmarkMeasurement) or not isinstance(scores, list):
            raise GatewayError(503, "research_measurement_failed")
        return measurement, scores
    except GatewayError:
        raise
    except Exception:
        raise GatewayError(503, "research_measurement_failed") from None
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
