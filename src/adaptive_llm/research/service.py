"""Isolated local studies and fine-tuning, with ordinary signed candidate publication."""

import asyncio
import copy
import hashlib
import json
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter

from adaptive_llm.contracts import (
    AdapterConfig,
    DatasetLineage,
    DatasetManifest,
    EvaluationReport,
    LifecycleTransition,
    ModelManifest,
    PruningLineage,
    TrainingJob,
    TrainingJobSpecification,
    now,
    uid,
)
from adaptive_llm.datasets.artifacts import read_shards, verify_manifest
from adaptive_llm.distillation.benchmark import LocalBenchmarker
from adaptive_llm.distillation.training import StudentTrainer
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.runner import isolated_persistence
from adaptive_llm.evaluation.service import EvaluationDeployment, LocalEvaluator
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.providers import ProviderRequest, ProviderResult
from adaptive_llm.registry.artifacts import artifact_digest, signed_metadata, verify_artifact
from adaptive_llm.research.models import (
    ActivationSpecification,
    BaselineSpecification,
    BaseSpecification,
    PruneSpecification,
    StudySummary,
)
from adaptive_llm.research.store import LocalStudyStore, StudyStore, authorize, markdown
from adaptive_llm.research.tensors import geometry, instrument, prune
from adaptive_llm.routing import Deployment, PriceList
from adaptive_llm.signing import sign_record
from adaptive_llm.training.lora import (
    BaseFiles,
    LoraGenerator,
    base_files,
    cpu,
    libraries,
    load_base,
)
from adaptive_llm.training.service import TrainingOrchestrator


class BaseGenerator(LoraGenerator):
    def __init__(self, manifest: ModelManifest, base: BaseFiles) -> None:
        self.manifest, self.libs = manifest, libraries()
        with cpu(self.libs, manifest.seed):
            self.model, self.tokenizer = load_base(base, self.libs, "fp32")
            self.model.eval()
        self.limit = base.context_limit


class SourceProvider:
    def __init__(self, generator: LoraGenerator) -> None:
        self.generator = generator

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        return await asyncio.to_thread(self.generator.generate, request)


@dataclass
class SourceIdentification:
    base: BaseFiles
    manifest: ModelManifest
    files: dict[str, bytes]
    deployment: Deployment
    adapter_digest: str | None

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def artifact_digest(self) -> str:
        return self.adapter_digest or self.base.digest


@dataclass
class Source:
    base: BaseFiles
    generator: LoraGenerator
    deployment: EvaluationDeployment
    adapter_digest: str | None


class PrunedTrainer(StudentTrainer):
    def use_base(self, base: BaseFiles) -> None:
        self.snapshot = base
        self.architecture = "pruned-full-v1"

    def _base(self, spec: TrainingJobSpecification) -> BaseFiles:
        return self.snapshot


def lineage(dataset: DatasetManifest) -> DatasetLineage:
    return DatasetLineage(
        dataset_id=dataset.dataset_id,
        version=dataset.version,
        content_digest=dataset.content_digest,
    )


class ResearchService:
    def __init__(
        self,
        training: TrainingOrchestrator,
        evaluator: LocalEvaluator,
        benchmarker: LocalBenchmarker,
        licences_path: Path,
        memory_limit_bytes: int,
        time_limit_seconds: float,
        store: StudyStore | None = None,
    ) -> None:
        self.training, self.evaluator, self.benchmarker = training, evaluator, benchmarker
        self.data_dir, self.keyring, self.cipher = (
            training.data_dir,
            training.keyring,
            training.cipher,
        )
        self.root = self.data_dir / "research"
        self.store = store or LocalStudyStore(self.root, self.cipher, self.keyring)
        self.licences_path = licences_path
        self.memory_limit_bytes, self.time_limit_seconds = memory_limit_bytes, time_limit_seconds
        self.lock = RLock()

    def dataset(
        self, dataset_id: str, version: str, identity: Identity, *, verify_shards: bool = True
    ) -> DatasetManifest:
        authorize(identity)
        manifest = self.training.builder.get(dataset_id, version, identity)
        if verify_shards:
            read_shards(manifest, self.data_dir, self.cipher, self.keyring)
        else:
            verify_manifest(manifest, self.data_dir, self.keyring)
        if manifest.approval.status != "approved":
            raise GatewayError(409, "dataset_approval_required")
        if manifest.purpose != "adapter_training":
            raise GatewayError(409, "training_dataset_required")
        for tenant in manifest.tenant_ids:
            trusted = replace(identity, tenant_id=tenant, application_ids=frozenset({"training"}))
            decision = self.training.policy.decide(trusted, "training")
            if not decision.processing_allowed or not decision.training_allowed:
                raise GatewayError(403, "training_policy_denied")
        return manifest

    def identify_source(self, spec: BaseSpecification, identity: Identity) -> SourceIdentification:
        """Verify source bytes and bindings without allocating a model or importing tensors."""
        authorize(identity)
        base = base_files(
            self.data_dir,
            spec.base_model_id,
            spec.base_model_revision,
            spec.tokenizer_id,
            spec.chat_template_version,
            spec.base_model_licence,
        )
        if spec.base_model_licence not in json.loads(self.licences_path.read_text())["allowed"]:
            raise GatewayError(403, "research_licence_denied")
        geometry(base)
        if base.parameter_count * 24 > self.memory_limit_bytes:
            raise GatewayError(422, "training_memory_limit")
        adapter_digest = None
        files: dict[str, bytes] = {}
        if spec.adapter_version:
            manifest = self.training.registry.get(spec.adapter_version, identity)
            if (
                manifest.adapter_architecture not in {"lora-peft-v1", "student-full-v1"}
                or manifest.pruning is not None
                or manifest.state in {"revoked", "deprecated"}
                or any(
                    getattr(manifest, k) != getattr(spec, k)
                    for k in (
                        "base_model_id",
                        "base_model_revision",
                        "base_model_licence",
                        "tokenizer_id",
                        "chat_template_version",
                    )
                )
            ):
                raise GatewayError(409, "research_adapter_mismatch")
            files = verify_artifact(
                manifest, self.data_dir / manifest.storage_location, self.keyring
            )
            adapter_digest = manifest.artifact_digest
        else:
            # This manifest is an in-memory generation description, never a registered model.
            manifest = ModelManifest(
                registry_id="research-base",
                version="research-base-" + base.digest[:32],
                tenant_ids=[],
                **spec.model_dump(exclude={"schema_version", "adapter_version"}),
                adapter_config=AdapterConfig(),
                datasets=[],
                training_job_id=uid(),
                code_revision=self.training.revision,
                container_digest="local",
                configuration_digest=base.digest,
                seed=23,
                hardware_class="cpu",
                artifact_hashes={},
                artifact_digest=base.digest,
                artifact_mac="unregistered",
                storage_location="research",
                context_limit=base.context_limit,
            )
        foundation = self.evaluator.deployments[self.evaluator.foundation_id].manifest
        deployment = foundation.model_copy(
            update={
                "model_deployment_id": manifest.version,
                "model_version": base.digest + (adapter_digest or ""),
                "model_id": spec.base_model_id,
                "price_list": PriceList(
                    version=f"research-source-{manifest.version}",
                    input_micros_per_1000_tokens=manifest.input_micros_per_1000_tokens,
                    output_micros_per_1000_tokens=manifest.output_micros_per_1000_tokens,
                ),
            }
        )
        return SourceIdentification(base, manifest, files, deployment, adapter_digest)

    def load_source(self, source: SourceIdentification) -> Source:
        if source.adapter_digest is None:
            generator: LoraGenerator = BaseGenerator(source.manifest, source.base)
        else:
            generator = LoraGenerator(source.manifest, self.data_dir, source.files)
            if source.manifest.adapter_architecture == "lora-peft-v1":
                with cpu(generator.libs, source.manifest.seed):
                    generator.model = generator.model.merge_and_unload()
        return Source(
            source.base,
            generator,
            EvaluationDeployment(
                source.deployment, SourceProvider(generator), source.version, source.artifact_digest
            ),
            source.adapter_digest,
        )

    def baseline(self, request: BaselineSpecification, identity: Identity) -> EvaluationReport:
        self.dataset(request.evaluation.dataset_id, request.evaluation.dataset_version, identity)
        source = self.load_source(self.identify_source(request.base, identity))
        evaluator = copy.copy(self.evaluator)
        evaluator.deployments = {
            **evaluator.deployments,
            source.deployment.version: source.deployment,
        }
        spec = request.evaluation.model_copy(
            update={"candidate_deployment_id": source.deployment.version}
        )
        return evaluator.evaluate(spec, identity)

    def preconditions(
        self,
        spec: ActivationSpecification,
        identity: Identity,
        *,
        verify_shards: bool = True,
    ) -> tuple[SourceIdentification, DatasetManifest, DatasetManifest]:
        calibration = self.dataset(
            spec.calibration_dataset_id,
            spec.calibration_dataset_version,
            identity,
            verify_shards=verify_shards,
        )
        evaluation = (
            calibration
            if (
                spec.evaluation_dataset_id == spec.calibration_dataset_id
                and spec.evaluation_dataset_version == spec.calibration_dataset_version
            )
            else self.dataset(
                spec.evaluation_dataset_id,
                spec.evaluation_dataset_version,
                identity,
                verify_shards=verify_shards,
            )
        )
        source = self.identify_source(spec.base, identity)
        try:
            report = self.evaluator.get(spec.baseline_evaluation_id, identity)
        except GatewayError:
            raise GatewayError(409, "research_baseline_required") from None
        if (
            not report.passed
            or not all(g.passed for g in decisions(report))
            or report.candidate_manifest_version != source.version
            or report.candidate_artifact_digest != source.artifact_digest
            or report.specification.dataset_id != evaluation.dataset_id
            or report.specification.dataset_version != evaluation.version
            or report.dataset_content_digest != evaluation.content_digest
        ):
            raise GatewayError(409, "research_baseline_required")
        if spec.max_sequence_length > source.base.context_limit:
            raise GatewayError(422, "training_configuration_invalid")
        return source, calibration, evaluation

    def study(self, spec: ActivationSpecification, identity: Identity) -> StudySummary:
        with self.lock:
            start = perf_counter()
            source, calibration, evaluation = self.preconditions(spec, identity)
            try:
                previous, _ = self.store.get(spec.study_id, identity)
            except GatewayError as error:
                if error.code != "study_not_found":
                    raise
            else:
                if previous.specification != spec:
                    raise GatewayError(409, "study_id_conflict")
                return self.get(spec.study_id, identity)
            rows = read_shards(calibration, self.data_dir, self.cipher, self.keyring)[
                spec.calibration_split
            ]
            if not rows:
                raise GatewayError(409, "empty_calibration_split")
            rows = random.Random(spec.seed).sample(rows, min(spec.sample_size, len(rows)))
            loaded = self.load_source(source)
            # Allocate a fresh private connection just like evaluation; no tenant case writes.
            with (
                isolated_persistence(self.evaluator.persistence),
                cpu(loaded.generator.libs, spec.seed),
            ):
                raw, shapes, rankings = instrument(
                    loaded.generator.model,
                    loaded.generator.tokenizer,
                    rows,
                    spec.max_sequence_length,
                    loaded.generator.libs,
                    memory_limit_bytes=self.memory_limit_bytes,
                )
            if perf_counter() - start > self.time_limit_seconds:
                raise GatewayError(503, "research_time_limit")
            self.preconditions(spec, identity, verify_shards=False)
            summary = StudySummary(
                specification=spec,
                tenant_ids=sorted(set(calibration.tenant_ids + evaluation.tenant_ids)),
                base_digest=source.base.digest,
                adapter_digest=source.adapter_digest,
                calibration_dataset=lineage(calibration),
                evaluation_dataset=lineage(evaluation),
                sample_size=len(rows),
                wall_seconds=perf_counter() - start,
                shapes=shapes,
                rankings=rankings,
                parameter_count_before=source.base.parameter_count,
            )
            self.store.save(summary, raw, identity)
            return summary

    def get(self, study_id: str, identity: Identity) -> StudySummary:
        summary, _ = self.store.get(study_id, identity)
        models = [
            m
            for m in self.training.registry.models(identity)
            if m.pruning is not None and m.pruning.study_id == study_id
        ]
        evaluations = sorted({e for m in models for e in m.evaluation_reports})
        benchmarks = sorted(
            {
                b.specification.benchmark_id
                for m in models
                for e in m.evaluation_reports
                for b in self.benchmarker.store.for_evaluation(m.version, e, identity)
            }
        )
        result = summary.model_copy(
            update={
                "candidates": {
                    m.version: m.pruning.parameter_count_after for m in models if m.pruning
                },
                "evaluation_ids": evaluations,
                "benchmark_ids": benchmarks,
            }
        )
        if isinstance(self.store, LocalStudyStore):
            self.dataset(
                summary.calibration_dataset.dataset_id,
                summary.calibration_dataset.version,
                identity,
                verify_shards=False,
            )
            self.dataset(
                summary.evaluation_dataset.dataset_id,
                summary.evaluation_dataset.version,
                identity,
                verify_shards=False,
            )
            (self.store.path(study_id) / "report.md").write_text(markdown(result))
        return result

    def structured_prune(self, request: PruneSpecification, identity: Identity) -> ModelManifest:
        authorize(identity)
        if request.plan.ranking_rule != "ablation_sensitivity":
            raise GatewayError(422, "magnitude_ranking_unsupported")
        with self.lock:
            summary, _ = self.store.get(request.study_id, identity)
            if (
                "layers" in request.plan.structures
                and summary.hook_version != "aggregate-residual-taylor-v2"
            ):
                raise GatewayError(409, "research_hook_version_mismatch")
            source, calibration, evaluation = self.preconditions(summary.specification, identity)
            if (
                source.base.digest != summary.base_digest
                or source.adapter_digest != summary.adapter_digest
                or lineage(calibration) != summary.calibration_dataset
                or lineage(evaluation) != summary.evaluation_dataset
            ):
                raise GatewayError(409, "study_lineage_changed")
            spec = request.training.model_copy(update={"code_revision": self.training.revision})
            if (
                spec.job_type != "adapter"
                or spec.hardware_class != "cpu"
                or spec.dataset_id != calibration.dataset_id
                or spec.dataset_version != calibration.version
                or any(
                    getattr(spec, k) != getattr(summary.specification.base, k)
                    for k in (
                        "base_model_id",
                        "base_model_revision",
                        "base_model_licence",
                        "tokenizer_id",
                        "chat_template_version",
                    )
                )
            ):
                raise GatewayError(422, "research_training_mismatch")
            prior = self.training.registry.job(spec.job_id, identity)
            if prior is not None:
                if prior.state != "succeeded":
                    raise GatewayError(409, "training_job_id_conflict")
                model = self.training.registry.get(prior.model_version, identity)
                if (
                    prior.specification != spec
                    or model.pruning is None
                    or model.pruning.study_id != request.study_id
                    or model.pruning.plan != request.plan
                ):
                    raise GatewayError(409, "training_job_id_conflict")
                return model
            start = perf_counter()
            loaded = self.load_source(source)
            with cpu(loaded.generator.libs, spec.seed):
                compact, removed = prune(
                    loaded.generator.model,
                    source.base,
                    request.plan,
                    summary.rankings,
                    loaded.generator.libs,
                )
            trainer = PrunedTrainer(
                self.data_dir,
                self.cipher,
                self.keyring,
                full=True,
                memory_limit_bytes=self.memory_limit_bytes,
                time_limit_seconds=self.time_limit_seconds,
            )
            trainer.use_base(compact)
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            job = TrainingJob(specification=spec, trainer_architecture=trainer.architecture)
            destination = self.root / "models" / job.model_version
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with TemporaryDirectory(prefix=".prune-", dir=self.root) as temporary:
                working = Path(temporary) / "export"
                working.mkdir()
                with isolated_persistence(self.evaluator.persistence):
                    usage = trainer.train(
                        spec, calibration, working, [], lambda _: None, lambda: None
                    )
                for path in working.glob("checkpoint-*"):
                    shutil.rmtree(path)
                weights = (working / "merged.safetensors").read_bytes()
                (working / "adapter.safetensors").unlink()
                (working / "merged.safetensors").unlink()
                for name, content in compact.files.items():
                    path = working / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(weights if name == "model.safetensors" else content)
                self.preconditions(summary.specification, identity, verify_shards=False)
                hashes = {
                    p.relative_to(working).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in working.rglob("*")
                    if p.is_file()
                }
                model = ModelManifest(
                    registry_id=spec.registry_id,
                    version=job.model_version,
                    tenant_ids=summary.tenant_ids,
                    **summary.specification.base.model_dump(
                        exclude={"schema_version", "adapter_version"}
                    ),
                    adapter_architecture=trainer.architecture,
                    adapter_config=spec.adapter_config,
                    datasets=[lineage(calibration)],
                    training_job_id=spec.job_id,
                    code_revision=spec.code_revision,
                    container_digest=spec.container_digest,
                    configuration_digest=hashlib.sha256(
                        request.model_dump_json().encode()
                    ).hexdigest(),
                    seed=spec.seed,
                    hardware_class="cpu",
                    context_limit=spec.max_sequence_length,
                    input_micros_per_1000_tokens=spec.input_micros_per_1000_tokens,
                    output_micros_per_1000_tokens=spec.output_micros_per_1000_tokens,
                    artifact_hashes=hashes,
                    artifact_digest=artifact_digest(hashes),
                    artifact_mac="pending",
                    manifest_mac_version="2",
                    storage_location=destination.relative_to(self.data_dir).as_posix(),
                    capability_signature_version="1",
                    safety_notes=["Requires full evaluation and hardware benchmark."],
                    known_limitations=[
                        "Local tiny research; no production acceptance.",
                        "GQA expansion may offset attention parameter savings.",
                    ],
                    lifecycle_history=[
                        LifecycleTransition(
                            from_state=None,
                            to_state="candidate",
                            actor=identity.subject_id_pseudonymous or "operator",
                            reason="structured_prune_completed",
                        )
                    ],
                    pruning=PruningLineage(
                        study_id=request.study_id,
                        plan=request.plan,
                        removed_indices=removed,
                        parameter_count_before=source.base.parameter_count,
                        parameter_count_after=compact.parameter_count,
                        base_digest=source.base.digest,
                        adapter_version=summary.specification.base.adapter_version,
                        adapter_digest=source.adapter_digest,
                        calibration_dataset=lineage(calibration),
                        evaluation_dataset=lineage(evaluation),
                        baseline_evaluation_id=summary.specification.baseline_evaluation_id,
                    ),
                )
                self.keyring.require_signer()
                if self.keyring.signer is not None:
                    model = model.model_copy(update={"artifact_mac": ""})
                    model = sign_record(
                        model, self.keyring.signer, "model-manifest", signed_metadata(model)
                    )
                else:
                    model = model.model_copy(
                        update={"artifact_mac": self.keyring.artifact_mac(signed_metadata(model))}
                    )
                verify_artifact(model, working, self.keyring)
                working.rename(destination)
                try:
                    self.training.registry.complete(
                        job.model_copy(
                            update={
                                "state": "succeeded",
                                "started_at": job.created_at,
                                "completed_at": now(),
                                "resource_usage": usage.model_copy(
                                    update={
                                        "wall_seconds": perf_counter() - start,
                                        "artifact_bytes": sum(
                                            (destination / k).stat().st_size for k in hashes
                                        ),
                                    }
                                ),
                                "artifact_ref": model.storage_location,
                                "artifact_digest": model.artifact_digest,
                            }
                        ),
                        model,
                        trainer_architecture=trainer.architecture,
                    )
                except Exception:
                    destination.rename(working)
                    raise
            self.get(request.study_id, identity)
            return model
