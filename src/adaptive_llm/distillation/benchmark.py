"""Paired local deployment measurements with authenticated, content-free control records."""

import asyncio
import hashlib
import hmac
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Protocol

from adaptive_llm.contracts import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSpecification,
    EvaluationReport,
    ModelManifest,
    PairedComparison,
)
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.distillation.data import teacher_deployment, teacher_digest
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.runner import Case, Runner, isolated_persistence
from adaptive_llm.evaluation.service import EvaluationDeployment, LocalEvaluator
from adaptive_llm.evaluation.stats import paired_bootstrap, percentile
from adaptive_llm.evaluation.suites.scoring import citations, token_f1
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.storage.sqlite import SQLiteDatabase
from adaptive_llm.training.lora import rss_bytes


def targets() -> dict[str, float]:
    path = Path(__file__).resolve().parents[3] / "configs/evaluation/initial-targets.json"
    values: dict[str, float] = json.loads(path.read_text())["distillation"]
    return values


def passes(report: BenchmarkReport) -> bool:
    thresholds = targets()
    quality = report.quality_comparison
    return bool(
        quality.sample_size >= thresholds["minimum_samples"]
        and quality.ci_lower is not None
        and quality.ci_lower > -thresholds["non_inferiority_margin"]
        and report.student.requests == report.teacher.requests
        and report.student.successes > 0
        and report.teacher.successes > 0
        and (
            (
                report.latency_reduction_fraction is not None
                and report.latency_reduction_fraction >= thresholds["p95_latency_reduction_min"]
            )
            or (
                report.cost_reduction_fraction is not None
                and report.cost_reduction_fraction >= thresholds["cost_reduction_min"]
            )
        )
    )


class BenchmarkStore(Protocol):
    def get(self, benchmark_id: str, identity: Identity) -> BenchmarkReport | None: ...

    def publish(self, report: BenchmarkReport, identity: Identity) -> BenchmarkReport: ...

    def for_evaluation(
        self,
        candidate: str,
        evaluation: str,
        identity: Identity,
    ) -> list[BenchmarkReport]: ...


class SQLiteBenchmarkStore:
    def __init__(self, database: SQLiteDatabase, keyring: Keyring) -> None:
        self.database, self.keyring = database, keyring

    def _mac(self, report: BenchmarkReport) -> str:
        return self.keyring.report_mac("benchmark-v1:" + report.model_dump_json(exclude={"mac"}))

    def get(self, benchmark_id: str, identity: Identity) -> BenchmarkReport | None:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT report FROM benchmark_reports WHERE benchmark_id=?",
                (benchmark_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            report = BenchmarkReport.model_validate_json(row[0])
            if report.specification.benchmark_id != benchmark_id or not hmac.compare_digest(
                report.mac, self._mac(report)
            ):
                raise ValueError
        except Exception:
            raise GatewayError(409, "benchmark_integrity_failed") from None
        if not set(report.tenant_ids) <= identity.dataset_tenants:
            return None
        return report

    def publish(self, report: BenchmarkReport, identity: Identity) -> BenchmarkReport:
        LocalDatasetBuilder._authorize(identity, report.tenant_ids)
        signed = report.model_copy(update={"mac": self._mac(report)})
        with self.database.transaction():
            if self.database.connection.execute(
                "SELECT 1 FROM benchmark_reports WHERE benchmark_id=?",
                (signed.specification.benchmark_id,),
            ).fetchone():
                raise GatewayError(409, "benchmark_exists")
            self.database.connection.execute(
                "INSERT INTO benchmark_reports VALUES (?, ?, ?, ?)",
                (
                    signed.specification.benchmark_id,
                    signed.specification.candidate_version,
                    signed.specification.evaluation_id,
                    signed.model_dump_json(),
                ),
            )
        return signed

    def for_evaluation(
        self,
        candidate: str,
        evaluation: str,
        identity: Identity,
    ) -> list[BenchmarkReport]:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            ids = self.database.connection.execute(
                "SELECT benchmark_id FROM benchmark_reports "
                "WHERE candidate_version=? AND evaluation_id=?",
                (candidate, evaluation),
            ).fetchall()
        return [
            r
            for row in ids
            if (r := self.get(row[0], identity)) is not None
            and r.specification.candidate_version == candidate
            and r.specification.evaluation_id == evaluation
        ]


class Benchmarker(Protocol):
    def run(self, spec: BenchmarkSpecification, identity: Identity) -> BenchmarkReport: ...

    def get(self, benchmark_id: str, identity: Identity) -> BenchmarkReport: ...


class LocalBenchmarker:
    def __init__(self, evaluator: LocalEvaluator, store: BenchmarkStore) -> None:
        self.evaluator, self.store = evaluator, store

    def get(self, benchmark_id: str, identity: Identity) -> BenchmarkReport:
        report = self.store.get(benchmark_id, identity)
        if report is None:
            raise GatewayError(404, "benchmark_not_found")
        return report

    def allows(
        self, model: ModelManifest, evaluation: EvaluationReport, identity: Identity
    ) -> bool:
        lineage = model.distillation
        if lineage is None or evaluation.distillation != lineage:
            return False
        teacher = teacher_deployment(self.evaluator, lineage.teacher_deployment_id, identity)
        if (
            teacher.version != lineage.teacher_version
            or teacher_digest(teacher) != lineage.teacher_artifact_digest
        ):
            return False
        return any(
            report.passed
            and passes(report)
            and report.specification.candidate_version == model.version
            and report.specification.evaluation_id == evaluation.specification.evaluation_id
            and report.candidate_artifact_digest == model.artifact_digest
            and report.teacher_version == lineage.teacher_version
            and report.teacher_artifact_digest == lineage.teacher_artifact_digest
            and report.dataset_content_digest == evaluation.dataset_content_digest
            for report in self.store.for_evaluation(
                model.version, evaluation.specification.evaluation_id, identity
            )
        )

    def run(self, spec: BenchmarkSpecification, identity: Identity) -> BenchmarkReport:
        LocalDatasetBuilder._authorize(identity, [])
        if identity.environment != "local":
            raise GatewayError(403, "benchmark_local_only")
        existing = self.store.get(spec.benchmark_id, identity)
        if existing is not None:
            if existing.specification != spec:
                raise GatewayError(409, "benchmark_id_conflict")
            return existing
        evaluator = self.evaluator
        if evaluator.registry is None:
            raise GatewayError(409, "benchmark_registry_required")
        model = evaluator.registry.get(spec.candidate_version, identity)
        evaluation = evaluator.get(spec.evaluation_id, identity)
        lineage = model.distillation
        if (
            lineage is None
            or evaluation.distillation != lineage
            or evaluation.candidate_manifest_version != model.version
            or evaluation.candidate_artifact_digest != model.artifact_digest
        ):
            raise GatewayError(409, "benchmark_evaluation_mismatch")
        teacher = teacher_deployment(evaluator, lineage.teacher_deployment_id, identity)
        if (
            teacher.version != lineage.teacher_version
            or teacher_digest(teacher) != lineage.teacher_artifact_digest
        ):
            raise GatewayError(409, "distillation_teacher_changed")
        candidate = evaluator._deployment(model.version, identity)
        manifest, cases = evaluator.reader.read(evaluation.specification, identity)
        if spec.requests < len(cases) or not cases:
            raise GatewayError(422, "benchmark_coverage_required")
        with isolated_persistence(evaluator.persistence) as scratch:
            student_runner = evaluator._runner(model.version, identity, scratch)
            teacher_runner = evaluator._runner(lineage.teacher_deployment_id, identity, scratch)

            async def run_pair() -> tuple[
                BenchmarkMeasurement, BenchmarkMeasurement, PairedComparison
            ]:
                # Warm both deployments; training reports cold load resource usage.
                await student_runner.run(cases[0])
                await teacher_runner.run(cases[0])
                student, left = await measure(student_runner, candidate, cases, spec.requests)
                teacher_measurement, right = await measure(
                    teacher_runner, teacher, cases, spec.requests
                )
                comparison = paired_bootstrap(
                    left,
                    right,
                    confidence=evaluation.specification.confidence_level,
                    seed=evaluation.specification.seed,
                )
                return student, teacher_measurement, comparison

            student, teacher_result, comparison = asyncio.run(run_pair())
        latency = (
            1 - student.p95_latency_ms / teacher_result.p95_latency_ms
            if teacher_result.p95_latency_ms > 0
            else None
        )
        cost = (
            1
            - (student.total_cost_micros / student.successes)
            / (teacher_result.total_cost_micros / teacher_result.successes)
            if student.successes and teacher_result.successes and teacher_result.total_cost_micros
            else None
        )
        report = BenchmarkReport(
            specification=spec,
            tenant_ids=manifest.tenant_ids,
            candidate_artifact_digest=model.artifact_digest,
            teacher_version=lineage.teacher_version,
            teacher_artifact_digest=lineage.teacher_artifact_digest,
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
            student=student,
            teacher=teacher_result,
            quality_comparison=comparison,
            latency_reduction_fraction=latency,
            cost_reduction_fraction=cost,
            passed=False,
            known_limitations=[
                "Local synthetic request mix; token prices are configured, not measured invoices.",
                "Warm concurrency 4; tensor execution is serialized by the shared CPU lock.",
                "Peak RSS is the process high-water mark including both models, not attribution.",
                "Latency varies with host scheduling; quality and costs are deterministic.",
            ],
        )
        report = report.model_copy(
            update={
                "passed": evaluation.passed
                and all(g.passed for g in decisions(evaluation))
                and passes(report)
            }
        )
        return self.store.publish(report, identity)


async def measure(
    runner: Runner,
    deployment: EvaluationDeployment,
    cases: list[Case],
    requests: int,
) -> tuple[BenchmarkMeasurement, list[float]]:
    semaphore = asyncio.Semaphore(4)

    async def one(index: int) -> tuple[int, float, int, bool, float]:
        async with semaphore:
            case = cases[index % len(cases)]
            outcome = await runner.run(case)
            precision, recall = citations(case, outcome)
            score = min(
                precision,
                recall,
                token_f1(outcome.response.content, case.target)
                if outcome.response is not None
                else 0,
            )
            # PipelineRunner returns a response only after generation and hard validation pass.
            # Quality is measured separately by the paired scores and the evaluation gates.
            success = outcome.response is not None and not outcome.error
            return index % len(cases), outcome.latency_ms, outcome.cost_micros, success, score

    start = perf_counter()
    rows = await asyncio.gather(*(one(i) for i in range(requests)))
    elapsed = perf_counter() - start
    scores: dict[int, list[float]] = defaultdict(list)
    for index, _, _, _, score in rows:
        scores[index].append(score)
    successes = sum(r[3] for r in rows)
    total = sum(r[2] for r in rows)
    prices = deployment.manifest.price_list
    return BenchmarkMeasurement(
        requests=requests,
        successes=successes,
        p50_latency_ms=percentile([r[1] for r in rows], 0.5),
        p95_latency_ms=percentile([r[1] for r in rows], 0.95),
        requests_per_second=requests / max(elapsed, 1e-9),
        peak_rss_bytes=rss_bytes(),
        total_cost_micros=total,
        cost_per_success_micros=(total + successes - 1) // successes if successes else None,
        input_micros_per_1000_tokens=prices.input_micros_per_1000_tokens,
        output_micros_per_1000_tokens=prices.output_micros_per_1000_tokens,
    ), [fmean(scores[i]) for i in range(len(cases))]
