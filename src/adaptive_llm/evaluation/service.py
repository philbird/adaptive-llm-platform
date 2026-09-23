"""Local evaluation orchestration; compute outside SQLite transactions and the HTTP loop."""

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import stdev
from typing import Protocol

from adaptive_llm.contracts import (
    EvaluationInput,
    EvaluationReport,
    EvaluationSpecification,
    ItemScore,
    PairedComparison,
    SuiteName,
    SuiteResult,
    now,
)
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.evaluation.data import DatasetReader, fixture_cases
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.judge import DeterministicJudge, Judge, validate_versions
from adaptive_llm.evaluation.runner import Case, PipelineRunner, Runner, isolated_persistence
from adaptive_llm.evaluation.stats import paired_bootstrap, required_sample_size
from adaptive_llm.evaluation.storage import EvaluationStore
from adaptive_llm.evaluation.suites import Suite
from adaptive_llm.evaluation.suites.golden import GoldenSuite
from adaptive_llm.evaluation.suites.held_out import HeldOutSuite
from adaptive_llm.evaluation.suites.performance import PerformanceSuite
from adaptive_llm.evaluation.suites.retrieval import RetrievalSuite
from adaptive_llm.evaluation.suites.safety import SafetySuite
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.providers import Provider
from adaptive_llm.routing import Deployment, FoundationRouter
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.validation import Validator

PILOT_DELTAS = (-0.01, 0.0, 0.01)
PILOT_SD = stdev(PILOT_DELTAS)


@dataclass(frozen=True)
class EvaluationDeployment:
    manifest: Deployment
    provider: Provider

    @property
    def version(self) -> str:
        return hashlib.sha256(self.manifest.model_dump_json().encode()).hexdigest()


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationInput, identity: Identity) -> EvaluationReport: ...

    def get(self, evaluation_id: str, identity: Identity) -> EvaluationReport: ...


class LocalEvaluator:
    def __init__(
        self,
        persistence: Persistence,
        reader: DatasetReader,
        store: EvaluationStore,
        deployments: Mapping[str, EvaluationDeployment],
        foundation_id: str,
        fixture_dir: Path,
        routing_path: Path,
        validator: Validator,
        revision: str,
        judge: Judge | None = None,
    ) -> None:
        self.persistence, self.reader, self.store = persistence, reader, store
        self.deployments, self.foundation_id = dict(deployments), foundation_id
        self.fixture_dir, self.routing_path = fixture_dir, routing_path
        self.validator, self.revision = validator, revision
        self.judge = judge or DeterministicJudge()
        self.suites: dict[SuiteName, Suite] = {
            "golden": GoldenSuite(self.judge),
            "held_out": HeldOutSuite(),
            "safety": SafetySuite(),
            "retrieval": RetrievalSuite(),
            "performance": PerformanceSuite(),
        }

    def get(self, evaluation_id: str, identity: Identity) -> EvaluationReport:
        report = self.store.get(evaluation_id, identity)
        if report is None:
            raise GatewayError(404, "evaluation_not_found")
        return report

    def _runner(self, deployment_id: str, identity: Identity, persistence: Persistence) -> Runner:
        deployment = self.deployments.get(deployment_id)
        if deployment is None:
            raise GatewayError(422, "unknown_evaluation_deployment")
        router = FoundationRouter(self.routing_path)
        router.deployment = deployment.manifest
        return PipelineRunner(
            persistence,
            identity,
            deployment.provider,
            router,
            self.validator,
            deployment.manifest.price_list,
        )

    async def _run(
        self, spec: EvaluationSpecification, cases: dict[SuiteName, list[Case]], runner: Runner
    ) -> list[SuiteResult]:
        return [await self.suites[name].run(runner, cases[name], spec) for name in spec.suites]

    async def _run_comparison(
        self,
        spec: EvaluationSpecification,
        cases: dict[SuiteName, list[Case]],
        identity: Identity,
        persistence: Persistence,
        locked_results: list[SuiteResult] | None,
    ) -> tuple[list[SuiteResult], list[SuiteResult]]:
        candidate = await self._run(
            spec, cases, self._runner(spec.candidate_deployment_id, identity, persistence)
        )
        if locked_results is not None:
            baseline = locked_results
        elif spec.baseline_deployment_id is not None:
            baseline = await self._run(
                spec, cases, self._runner(spec.baseline_deployment_id, identity, persistence)
            )
        else:
            baseline = []
        return candidate, baseline

    def evaluate(self, request: EvaluationInput, identity: Identity) -> EvaluationReport:
        LocalDatasetBuilder._authorize(identity, [])
        if identity.environment != "local":
            raise GatewayError(403, "evaluation_local_only")
        spec = EvaluationSpecification.model_validate(
            request.model_dump(exclude={"replace", "operator_note"})
        )
        existing = self.store.get(spec.evaluation_id, identity)
        if existing:
            if existing.specification != spec:
                raise GatewayError(409, "evaluation_id_conflict")
            return existing
        validate_versions(self.judge, spec.judge_version, spec.rubric_version)
        start = now()
        manifest, held_out = self.reader.read(spec, identity)
        if spec.candidate_deployment_id not in self.deployments or (
            spec.baseline_deployment_id is not None
            and spec.baseline_deployment_id not in self.deployments
        ):
            raise GatewayError(422, "unknown_evaluation_deployment")
        locking = spec.candidate_deployment_id == spec.baseline_deployment_id == self.foundation_id
        locked = (
            self.store.baseline(spec.baseline_deployment_id, spec.dataset_version, identity)
            if spec.baseline_deployment_id
            else None
        )
        if locking and locked and not request.replace:
            raise GatewayError(409, "baseline_already_locked")
        if request.replace and not locking:
            raise GatewayError(422, "replacement_requires_foundation")
        if not locking and spec.baseline_deployment_id is not None and not locked:
            raise GatewayError(409, "baseline_not_locked")
        tenant = sorted(manifest.tenant_ids)[0]
        paths = [
            self.fixture_dir / "golden/synthetic.jsonl",
            self.fixture_dir / "safety/synthetic.jsonl",
            self.fixture_dir / "retrieval/relevance.jsonl",
        ]
        digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
        golden, safety, retrieval = [fixture_cases(path, tenant) for path in paths]
        cases: dict[SuiteName, list[Case]] = {
            "golden": golden,
            "safety": safety,
            "retrieval": retrieval,
            "held_out": held_out,
            "performance": golden,
        }
        candidate_version = self.deployments[spec.candidate_deployment_id].version
        baseline_version = (
            self.deployments[spec.baseline_deployment_id].version
            if spec.baseline_deployment_id
            else None
        )
        if locked and not locking:
            pinned = locked.specification
            if (
                locked.dataset_content_digest != manifest.content_digest
                or locked.suite_content_digest != digest
                or locked.candidate_manifest_version != baseline_version
                or pinned.dataset_id != spec.dataset_id
                or pinned.rubric_version != spec.rubric_version
                or pinned.judge_version != spec.judge_version
                or pinned.critical_segments != spec.critical_segments
                or pinned.suites != spec.suites
                or pinned.seed != spec.seed
            ):
                raise GatewayError(409, "baseline_version_mismatch")
        with isolated_persistence(self.persistence) as scratch:
            candidate_results, baseline_results = asyncio.run(
                self._run_comparison(
                    spec,
                    cases,
                    identity,
                    scratch,
                    locked.suite_results if locked and not locking else None,
                )
            )
        candidate_scores = self._scores(candidate_results)
        baseline_scores = self._scores(baseline_results)
        comparison = self._compare(candidate_scores, baseline_scores, spec)
        segments = sorted({key for score in candidate_scores for key in score.segments})
        segment_comparisons = {
            key: self._compare(
                [s for s in candidate_scores if key in s.segments],
                [s for s in baseline_scores if key in s.segments],
                spec,
            )
            for key in segments
        }
        report = EvaluationReport(
            specification=spec,
            candidate_manifest_version=candidate_version,
            baseline_manifest_version=baseline_version,
            baseline_report_id=locked.specification.evaluation_id if locked else None,
            dataset_content_digest=manifest.content_digest,
            suite_content_digest=digest,
            code_revision=self.revision,
            started_at=start,
            completed_at=now(),
            suite_results=candidate_results,
            baseline_suite_results=baseline_results,
            paired_comparison=comparison,
            segment_comparisons=segment_comparisons,
            coverage={result.suite: result.items for result in candidate_results},
            pilot_standard_deviation=PILOT_SD,
            derived_minimum_sample_size=required_sample_size(
                spec.non_inferiority_margin, PILOT_SD, spec.confidence_level
            ),
            gate_decisions=[],
            passed=False,
            known_limitations=[
                "Local synthetic baseline; no production promotion or owner approval.",
                "Rule-based judge is not human calibrated; human review is out of scope.",
                "Paired quality is min(F1, citation precision, recall).",
                "Synthetic pilot deltas [-0.01, 0, 0.01], sample SD 0.01; not real workload.",
                "Held-out safety segments use risk labels; adversarial tests gate safety too.",
                "No streaming TTFT, cold-start or process-memory attribution; local latency only.",
                "Costs cover fake provider tokens; retrieval and infrastructure costs unavailable.",
                "Specialist cost reduction targets do not apply to synthetic foundation tests.",
                "Local MAC; asymmetric signing and external artifact lifecycle remain future work.",
                "Retrieval uses immutable source snapshots and ACL decoys; no live index.",
                "Latency includes private in-memory persistence; no tenant writes or case events.",
            ],
        )
        gates = decisions(report)
        report = report.model_copy(
            update={"gate_decisions": gates, "passed": all(g.passed for g in gates)}
        )
        self.store.publish(
            report,
            manifest.tenant_ids,
            identity,
            lock_baseline=locking and report.passed,
            expected_baseline=locked.specification.evaluation_id if locked else None,
            note=request.operator_note,
        )
        return report

    @staticmethod
    def _scores(results: list[SuiteResult]) -> list[ItemScore]:
        return next((result.scores for result in results if result.suite == "held_out"), [])

    @staticmethod
    def _compare(
        candidate: list[ItemScore], baseline: list[ItemScore], spec: EvaluationSpecification
    ) -> PairedComparison:
        left = {score.item_id: score for score in candidate}
        right = {score.item_id: score for score in baseline}
        if set(left) != set(right) or len(left) != len(candidate) or len(right) != len(baseline):
            return PairedComparison(sample_size=0)
        keys = sorted(left)
        return paired_bootstrap(
            [left[key].score for key in keys],
            [right[key].score for key in keys],
            confidence=spec.confidence_level,
            seed=spec.seed,
        )
