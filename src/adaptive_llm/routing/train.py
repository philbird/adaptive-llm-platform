"""Counterfactual numeric dataset construction and the ordinary signed training job backend."""

import json
from collections.abc import Callable
from pathlib import Path

from adaptive_llm.contracts import (
    DatasetManifest,
    DatasetSpecification,
    ResourceUsage,
    RoutingFeatures,
    RoutingObservation,
    RoutingRow,
    ShadowComparison,
    TrainingJobSpecification,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.datasets.builder import DatasetBuilder
from adaptive_llm.datasets.construction import BuiltExample
from adaptive_llm.datasets.eligibility import Example
from adaptive_llm.evaluation.storage import EvaluationStore
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.routing.model import train_model
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.database import Database
from adaptive_llm.training.fake import write_once


class CounterfactualRows:
    def __init__(
        self,
        builder: DatasetBuilder,
        control: Database,
        evaluations: EvaluationStore,
        data_dir: Path,
        cipher: PayloadCipher,
        keyring: Keyring,
        foundation_id: str,
    ) -> None:
        self.builder, self.control, self.evaluations = builder, control, evaluations
        self.data_dir, self.cipher, self.keyring = data_dir, cipher, keyring
        self.foundation_id = foundation_id

    def build(
        self,
        examples: list[Example],
        specification: DatasetSpecification,
        identity: Identity,
    ) -> list[BuiltExample]:
        assert specification.source_dataset_id and specification.source_dataset_version
        source = self.builder.get(
            specification.source_dataset_id, specification.source_dataset_version, identity
        )
        if source.purpose == "router_training" or not set(source.tenant_ids) <= set(
            specification.tenant_ids
        ):
            raise GatewayError(409, "routing_source_dataset_invalid")
        shards = read_shards(source, self.data_dir, self.cipher, self.keyring)
        held_out = {r["interaction_id"]: r for line in shards["test"] if (r := json.loads(line))}
        with self.control.lock:
            comparisons = [
                ShadowComparison.model_validate_json(r[0])
                for r in self.control.connection.execute("SELECT data FROM shadow_comparisons")
            ]
            reports = [
                r[0]
                for r in self.control.connection.execute(
                    "SELECT evaluation_id FROM evaluation_reports WHERE "
                    "json_extract(report, '$.specification.dataset_version')=? ORDER BY created_at",
                    (source.version,),
                )
            ]
        by_interaction: dict[str, list[ShadowComparison]] = {}
        selected_ids = {e.interaction.interaction_id for e in examples}
        for comparison in comparisons:
            if comparison.interaction_id in selected_ids:
                by_interaction.setdefault(comparison.interaction_id, []).append(comparison)
        scores: dict[tuple[str, str], RoutingObservation] = {}
        for report_id in reports:
            report = self.evaluations.get(report_id, identity)
            if report is None or report.dataset_content_digest != source.content_digest:
                continue
            for suite in report.suite_results:
                if suite.suite == "held_out":
                    for item in suite.scores:
                        scores[item.item_id, report.specification.candidate_deployment_id] = (
                            item.observation or RoutingObservation()
                        ).model_copy(update={"quality": item.score})
        versions = sorted(
            {c.specialist_version for values in by_interaction.values() for c in values}
            | {version for _, version in scores if version != self.foundation_id}
        )
        built: list[BuiltExample] = []
        for example in examples:
            interaction = example.interaction
            iid = interaction.interaction_id
            # All source held-out items survive missing specialist coverage. Shadow observations
            # extend that workload without selecting only successfully compared interactions.
            if iid not in held_out and iid not in by_interaction:
                with self.control.lock:
                    observed = self.control.connection.execute(
                        "SELECT 1 FROM shadow_observations WHERE interaction_id=?", (iid,)
                    ).fetchone()
                if observed is None:
                    continue
            retrieval = example.retrieval
            chunks = [c for c in retrieval.candidates if c.supplied_to_model] if retrieval else []
            attempt = example.attempt
            foundation = (
                RoutingObservation(
                    quality=None,
                    validation_pass=bool(
                        attempt and attempt.validation and attempt.validation.passed
                    ),
                    cost_micros=attempt.estimated_cost_micros if attempt else None,
                    latency_ms=attempt.total_latency_ms if attempt else 0,
                )
                if attempt and attempt.deployment_id == self.foundation_id
                else None
            )
            candidates: dict[str, RoutingObservation | None] = {v: None for v in versions}
            candidates[self.foundation_id] = foundation
            for comparison in by_interaction.get(iid, []):
                if comparison.tenant_id != interaction.tenant_id or foundation is None:
                    continue
                candidates[self.foundation_id] = foundation.model_copy(
                    update={"quality": comparison.foundation_score}
                )
                candidates[comparison.specialist_version] = RoutingObservation(
                    quality=comparison.specialist_score,
                    validation_pass=comparison.specialist_validation.passed,
                    cost_micros=max(0, foundation.cost_micros + comparison.cost_delta_micros)
                    if foundation.cost_micros is not None
                    else None,
                    latency_ms=max(0, foundation.latency_ms + comparison.latency_delta_ms)
                    if foundation.latency_ms is not None
                    else None,
                )
            original = held_out.get(iid, {})
            example_hash = original.get("example_hash")
            for version in candidates:
                if (example_hash, version) in scores:
                    candidates[version] = scores[example_hash, version]
            row = RoutingRow(
                interaction_id=iid,
                tenant_id=interaction.tenant_id,
                features=RoutingFeatures(
                    task=interaction.task.label,
                    language=interaction.task.language,
                    risk_tier=interaction.task.risk_tier,
                    chunk_count=len(chunks),
                    top_score=max((c.retrieval_score for c in chunks), default=0),
                    index_id=retrieval.index_id if retrieval else None,
                    input_tokens=attempt.usage.input_tokens
                    if attempt and attempt.usage
                    else interaction.input.token_count,
                    context_supplied=bool(chunks),
                ),
                foundation_id=self.foundation_id,
                candidates=candidates,
                source_dataset_id=source.dataset_id,
                source_dataset_version=source.version,
                source_example_hash=example_hash,
            )
            built.append(
                BuiltExample(
                    iid,
                    interaction.tenant_id,
                    interaction.subject_id_pseudonymous,
                    tuple(
                        original.get("document_families", sorted({c.document_id for c in chunks}))
                    ),
                    interaction.started_at,
                    "production_output",
                    interaction.task.language,
                    self.keyring.dataset_hash(iid),
                    "",
                    row.model_dump(mode="json"),
                )
            )
        return built


class RouterTrainer:
    architecture = "router-logistic-v1"

    def __init__(self, data_dir: Path, cipher: PayloadCipher, keyring: Keyring) -> None:
        self.data_dir, self.cipher, self.keyring = data_dir, cipher, keyring

    def train(
        self,
        specification: TrainingJobSpecification,
        dataset: DatasetManifest,
        directory: Path,
        checkpoint_refs: list[str],
        checkpoint: Callable[[str], None],
        check: Callable[[], None],
    ) -> ResourceUsage:
        rows = read_shards(dataset, self.data_dir, self.cipher, self.keyring)
        training = [RoutingRow.model_validate_json(r) for r in rows["train"]]
        calibration = [RoutingRow.model_validate_json(r) for r in rows["validation"]]
        model = train_model(training, calibration, check)
        check()
        write_once(directory / "router.json", model.model_dump_json().encode())
        return ResourceUsage(examples=len(training), steps=1200)
