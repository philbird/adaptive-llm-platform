"""Pruned candidates require measured hardware benefit in addition to every quality gate."""

import hashlib
import json
from pathlib import Path

from adaptive_llm.contracts import (
    BenchmarkReport,
    BenchmarkSpecification,
    EvaluationReport,
    ModelManifest,
)
from adaptive_llm.distillation.benchmark import LocalBenchmarker
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.stats import paired_bootstrap
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.registry.artifacts import verify_artifact
from adaptive_llm.research.measurement import MeasurementInput, isolated_measure
from adaptive_llm.research.models import BaseSpecification
from adaptive_llm.research.service import ResearchService
from adaptive_llm.research.store import authorize
from adaptive_llm.routing import PriceList


def targets() -> dict[str, float]:
    values: dict[str, float] = json.loads(
        (
            Path(__file__).resolve().parents[3] / "configs/evaluation/initial-targets.json"
        ).read_text()
    )["pruning"]
    return values


def hardware_benefit(report: BenchmarkReport) -> bool:
    thresholds = targets()
    return bool(
        (
            report.latency_reduction_fraction is not None
            and report.latency_reduction_fraction >= thresholds["p95_latency_reduction_min"]
        )
        or (
            report.peak_rss_reduction_fraction is not None
            and report.peak_rss_reduction_fraction >= thresholds["peak_rss_reduction_min"]
        )
    )


def passes(report: BenchmarkReport) -> bool:
    thresholds = targets()
    quality = report.quality_comparison
    return bool(
        report.pruning_study_id
        and quality.sample_size >= thresholds["minimum_samples"]
        and quality.ci_lower is not None
        and quality.ci_lower > -thresholds["non_inferiority_margin"]
        and report.student.successes > 0
        and report.teacher.successes > 0
        and report.student.requests == report.teacher.requests == report.specification.requests
        and report.student.parameter_count
        and report.teacher.parameter_count
        and hardware_benefit(report)
    )


def safety_passes(evaluation: EvaluationReport) -> bool:
    safety = [s for s in evaluation.suite_results if s.suite == "safety"]
    return bool(
        len(safety) == 1
        and safety[0].completed
        and safety[0].items > 0
        and safety[0].metrics.get("critical_failures") == 0
        and not any(safety[0].failures.values())
    )


class ResearchBenchmarker(LocalBenchmarker):
    def __init__(self, ordinary: LocalBenchmarker, research: ResearchService) -> None:
        super().__init__(ordinary.evaluator, ordinary.store)
        self.ordinary, self.research = ordinary, research

    def allows(
        self, model: ModelManifest, evaluation: EvaluationReport, identity: Identity
    ) -> bool:
        if model.adapter_architecture != "pruned-full-v1":
            return self.ordinary.allows(model, evaluation, identity)
        lineage = model.pruning
        if lineage is None or not safety_passes(evaluation):
            return False
        # The standard registry verifies artifacts before this callback. Recheck the source too.
        source = self.research.identify_source(
            BaseSpecification(
                **model.model_dump(
                    include={
                        "base_model_id",
                        "base_model_revision",
                        "base_model_licence",
                        "tokenizer_id",
                        "chat_template_version",
                    }
                ),
                adapter_version=lineage.adapter_version,
            ),
            identity,
        )
        if (
            source.base.digest != lineage.base_digest
            or source.adapter_digest != lineage.adapter_digest
        ):
            return False
        return any(
            r.passed
            and passes(r)
            and r.pruning_study_id == lineage.study_id
            and r.candidate_artifact_digest == model.artifact_digest
            and r.teacher_version == source.version
            and r.teacher_artifact_digest == source.artifact_digest
            and r.dataset_content_digest == evaluation.dataset_content_digest
            and r.student.parameter_count == lineage.parameter_count_after
            and r.teacher.parameter_count == lineage.parameter_count_before
            for r in self.store.for_evaluation(
                model.version, evaluation.specification.evaluation_id, identity
            )
        )

    def run(self, spec: BenchmarkSpecification, identity: Identity) -> BenchmarkReport:
        registry = self.evaluator.registry
        if registry is None:
            return self.ordinary.run(spec, identity)
        model = registry.get(spec.candidate_version, identity)
        if model.adapter_architecture != "pruned-full-v1":
            return self.ordinary.run(spec, identity)
        authorize(identity)
        existing = self.store.get(spec.benchmark_id, identity)
        if existing:
            if existing.specification != spec:
                raise GatewayError(409, "benchmark_id_conflict")
            return existing
        lineage = model.pruning
        evaluation = self.evaluator.get(spec.evaluation_id, identity)
        if (
            lineage is None
            or evaluation.candidate_manifest_version != model.version
            or evaluation.candidate_artifact_digest != model.artifact_digest
            or evaluation.dataset_content_digest != lineage.evaluation_dataset.content_digest
            or evaluation.specification.dataset_version != lineage.evaluation_dataset.version
            or evaluation.specification.dataset_id != lineage.evaluation_dataset.dataset_id
        ):
            raise GatewayError(409, "benchmark_evaluation_mismatch")
        for dataset in (lineage.calibration_dataset, lineage.evaluation_dataset):
            self.research.dataset(dataset.dataset_id, dataset.version, identity)
        source = self.research.identify_source(
            BaseSpecification(
                **model.model_dump(
                    include={
                        "base_model_id",
                        "base_model_revision",
                        "base_model_licence",
                        "tokenizer_id",
                        "chat_template_version",
                    }
                ),
                adapter_version=lineage.adapter_version,
            ),
            identity,
        )
        if (
            source.base.digest != lineage.base_digest
            or source.adapter_digest != lineage.adapter_digest
        ):
            raise GatewayError(409, "study_lineage_changed")
        evaluator = self.evaluator
        manifest, cases = evaluator.reader.read(evaluation.specification, identity)
        if not cases or spec.requests < len(cases):
            raise GatewayError(422, "benchmark_coverage_required")
        if model.state in {"candidate", "deprecated", "revoked"}:
            raise GatewayError(409, "model_not_evaluating")
        candidate = evaluator.deployments[evaluator.foundation_id].manifest.model_copy(
            update={
                "model_deployment_id": model.version,
                "model_version": f"{model.base_model_revision}.{model.artifact_digest}",
                "model_id": model.base_model_id,
                "processing_region": model.processing_region,
                "price_list": PriceList(
                    version=f"model-{model.version}",
                    input_micros_per_1000_tokens=model.input_micros_per_1000_tokens,
                    output_micros_per_1000_tokens=model.output_micros_per_1000_tokens,
                ),
            }
        )
        right, right_scores = isolated_measure(
            MeasurementInput(
                source.manifest,
                source.files,
                None if lineage.adapter_version else source.base,
                source.deployment,
                cases,
                identity,
                evaluator.validator,
                evaluator.routing_path,
                self.research.data_dir,
                spec.requests,
            ),
            self.research.time_limit_seconds,
        )
        left, left_scores = isolated_measure(
            MeasurementInput(
                model,
                verify_artifact(
                    model, self.research.data_dir / model.storage_location, self.research.keyring
                ),
                None,
                candidate,
                cases,
                identity,
                evaluator.validator,
                evaluator.routing_path,
                self.research.data_dir,
                spec.requests,
            ),
            self.research.time_limit_seconds,
        )
        comparison = paired_bootstrap(
            left_scores,
            right_scores,
            confidence=evaluation.specification.confidence_level,
            seed=evaluation.specification.seed,
        )
        left = left.model_copy(update={"parameter_count": lineage.parameter_count_after})
        right = right.model_copy(update={"parameter_count": lineage.parameter_count_before})
        report = BenchmarkReport(
            specification=spec,
            tenant_ids=manifest.tenant_ids,
            candidate_artifact_digest=model.artifact_digest,
            teacher_version=source.version,
            teacher_artifact_digest=source.artifact_digest,
            dataset_content_digest=manifest.content_digest,
            request_mix_digest=hashlib.sha256(
                json.dumps(
                    [
                        manifest.content_digest,
                        [cases[i % len(cases)].item_id for i in range(spec.requests)],
                        [c.request.max_output_tokens for c in cases],
                    ]
                ).encode()
            ).hexdigest(),
            student=left,
            teacher=right,
            quality_comparison=comparison,
            latency_reduction_fraction=1 - left.p95_latency_ms / right.p95_latency_ms
            if right.p95_latency_ms
            else None,
            peak_rss_reduction_fraction=1 - left.peak_rss_bytes / right.peak_rss_bytes,
            cost_reduction_fraction=None,
            pruning_study_id=lineage.study_id,
            passed=False,
            known_limitations=[
                "Local tiny CPU research; no production acceptance.",
                "Warm concurrency four; CPU tensor execution serialized.",
                "Peak RSS measured in separate fresh processes, including interpreter overhead.",
            ],
        )
        reasons = []
        if not hardware_benefit(report):
            reasons.append("no_hardware_benefit")
        elif not passes(report):
            reasons.append("benchmark_quality_failed")
        if not evaluation.passed or not all(g.passed for g in decisions(evaluation)):
            reasons.append("evaluation_failed")
        if not safety_passes(evaluation):
            reasons.append("critical_safety_failure")
        for dataset in (lineage.calibration_dataset, lineage.evaluation_dataset):
            self.research.dataset(
                dataset.dataset_id, dataset.version, identity, verify_shards=False
            )
        report = self.store.publish(
            report.model_copy(update={"passed": not reasons, "gate_reasons": reasons}), identity
        )
        self.research.get(lineage.study_id, identity)
        return report
