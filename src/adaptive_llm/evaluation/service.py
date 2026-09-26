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
    PromotionRequest,
    SuiteName,
    SuiteResult,
    now,
)
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.evaluation.data import DatasetReader, fixture_cases, fixture_digest
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
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.providers import LoopScopedProvider, Provider
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.routing import Deployment, FoundationRouter, PriceList
from adaptive_llm.routing.tasks import RulesClassifier, TaskClassifier
from adaptive_llm.storage.persistence import Persistence
from adaptive_llm.validation import Validator

PILOT_DELTAS = (-0.01, 0.0, 0.01)
PILOT_SD = stdev(PILOT_DELTAS)


@dataclass(frozen=True)
class EvaluationDeployment:
    manifest: Deployment
    provider: Provider
    registered_version: str | None = None
    artifact_digest: str | None = None

    @property
    def version(self) -> str:
        return (
            self.registered_version
            or hashlib.sha256(self.manifest.model_dump_json().encode()).hexdigest()
        )


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
        registry: ModelRegistry | None = None,
        data_dir: Path | None = None,
        classifier: TaskClassifier | None = None,
        policy: PolicyEngine | None = None,
    ) -> None:
        self.persistence, self.reader, self.store = persistence, reader, store
        self.deployments, self.foundation_id = dict(deployments), foundation_id
        self.fixture_dir, self.routing_path = fixture_dir, routing_path
        self.validator, self.revision = validator, revision
        self.judge = judge or DeterministicJudge()
        self.registry, self.data_dir = registry, data_dir
        self.classifier = classifier or RulesClassifier()
        self.policy = policy
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

    def _deployment(self, deployment_id: str, identity: Identity) -> EvaluationDeployment:
        deployment = self.deployments.get(deployment_id)
        if deployment is not None:
            return deployment
        if self.registry is None or self.data_dir is None:
            raise GatewayError(422, "unknown_evaluation_deployment")
        try:
            model = self.registry.get(deployment_id, identity)
        except GatewayError as error:
            if error.status_code == 404:
                raise GatewayError(422, "unknown_evaluation_deployment") from None
            raise
        if model.state in {"candidate", "deprecated", "revoked"}:
            raise GatewayError(409, "model_not_evaluating")
        provider = SpecialistProvider(
            model, self.data_dir / model.storage_location, self.persistence.keyring, self.data_dir
        )
        foundation = self.deployments[self.foundation_id].manifest
        return EvaluationDeployment(
            foundation.model_copy(
                update={
                    "model_deployment_id": model.version,
                    "model_version": provider.model_version,
                    "model_id": model.base_model_id,
                    "processing_region": model.processing_region,
                    "price_list": PriceList(
                        version=f"model-{model.version}",
                        input_micros_per_1000_tokens=model.input_micros_per_1000_tokens,
                        output_micros_per_1000_tokens=model.output_micros_per_1000_tokens,
                    ),
                }
            ),
            provider,
            model.version,
            model.artifact_digest,
        )

    def _runner(self, deployment_id: str, identity: Identity, persistence: Persistence) -> Runner:
        deployment = self._deployment(deployment_id, identity)
        router = FoundationRouter(self.routing_path)
        router.deployment = deployment.manifest
        return PipelineRunner(
            persistence,
            identity,
            deployment.provider,
            router,
            self.validator,
            deployment.manifest.price_list,
            classifier=self.classifier,
            residency=deployment.manifest.processing_region,
        )

    async def _run(
        self, spec: EvaluationSpecification, cases: dict[SuiteName, list[Case]], runner: Runner
    ) -> list[SuiteResult]:
        try:
            return [await self.suites[name].run(runner, cases[name], spec) for name in spec.suites]
        finally:
            # Evaluations run in a worker's temporary loop, independent of the HTTP loop.
            if isinstance(runner, PipelineRunner) and isinstance(
                runner.provider, LoopScopedProvider
            ):
                await runner.provider.aclose()

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
        if (
            spec.application_id != "evaluation"
            and spec.application_id not in identity.application_ids
        ):
            raise GatewayError(403, "application_forbidden")
        existing = self.store.get(spec.evaluation_id, identity)
        if existing:
            if existing.specification != spec:
                raise GatewayError(409, "evaluation_id_conflict")
            return existing
        if "routing" in spec.suites:
            return self._evaluate_router(request, spec, identity)
        validate_versions(self.judge, spec.judge_version, spec.rubric_version)
        start = now()
        if spec.fixture_tenant_id is not None:
            tenant = spec.fixture_tenant_id
            if tenant is None or tenant not in identity.dataset_tenants:
                raise GatewayError(403, "evaluation_tenant_forbidden")
            trusted = Identity(
                tenant,
                frozenset({spec.application_id}),
                identity.environment,
                identity.subject_id_pseudonymous,
            )
            permission = self.policy.decide(trusted, spec.application_id) if self.policy else None
            if (
                permission is None
                or not permission.processing_allowed
                or not permission.evaluation_allowed
            ):
                raise GatewayError(403, "evaluation_forbidden")
            paths = [
                self.fixture_dir / suite / f"{spec.fixture_set}.jsonl"
                for suite in ("golden", "safety")
            ]
            dataset_digest = fixture_digest(paths[:1])
            tenant_ids = [tenant]
            held_out: list[Case] = []
        else:
            manifest, held_out = self.reader.read(spec, identity)
            dataset_digest = manifest.content_digest
            tenant_ids = manifest.tenant_ids
            tenant = sorted(tenant_ids)[0]
            paths = [
                self.fixture_dir / suite / f"{spec.fixture_set or 'synthetic'}.jsonl"
                for suite in ("golden", "safety")
            ]
            if spec.fixture_set is None:
                paths.append(self.fixture_dir / "retrieval/relevance.jsonl")
        model = None
        if spec.candidate_deployment_id not in self.deployments and self.registry is not None:
            try:
                model = self.registry.get(spec.candidate_deployment_id, identity)
            except GatewayError as error:
                if error.status_code == 404:
                    raise GatewayError(422, "unknown_evaluation_deployment") from None
                raise
            if not any(
                d.dataset_id == spec.dataset_id
                and d.version == spec.dataset_version
                and d.content_digest == dataset_digest
                for d in (
                    [model.pruning.evaluation_dataset]
                    if model.pruning is not None
                    else model.datasets
                )
            ):
                raise GatewayError(409, "model_dataset_mismatch")
            if model.state == "candidate":
                self.registry.promote(
                    PromotionRequest(
                        model_version=model.version,
                        target_state="evaluating",
                        reason="evaluation_requested",
                    ),
                    identity,
                )
        candidate_deployment = self._deployment(spec.candidate_deployment_id, identity)
        baseline_deployment = (
            self._deployment(spec.baseline_deployment_id, identity)
            if spec.baseline_deployment_id
            else None
        )
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
        digest = (
            fixture_digest(paths)
            if spec.fixture_set is not None
            else hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
        )
        loaded = [fixture_cases(path, tenant, spec.application_id) for path in paths]
        golden, safety = loaded[:2]
        retrieval = loaded[2] if len(loaded) > 2 else held_out
        cases: dict[SuiteName, list[Case]] = {
            "golden": golden,
            "safety": safety,
            "retrieval": retrieval,
            "held_out": held_out,
            "performance": held_out
            if spec.fixture_set is not None and spec.fixture_tenant_id is None
            else golden,
        }
        candidate_version = candidate_deployment.version
        baseline_version = baseline_deployment.version if baseline_deployment else None
        if locked and not locking:
            pinned = locked.specification
            if (
                locked.dataset_content_digest != dataset_digest
                or locked.suite_content_digest != digest
                or locked.candidate_manifest_version != baseline_version
                or pinned.dataset_id != spec.dataset_id
                or pinned.rubric_version != spec.rubric_version
                or pinned.judge_version != spec.judge_version
                or pinned.critical_segments != spec.critical_segments
                or pinned.suites != spec.suites
                or pinned.seed != spec.seed
                or pinned.application_id != spec.application_id
                or pinned.fixture_set != spec.fixture_set
                or pinned.fixture_tenant_id != spec.fixture_tenant_id
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
            teacher_scores: list[ItemScore] | None = None
            if model is not None and model.distillation is not None:
                from adaptive_llm.distillation.data import teacher_deployment, teacher_digest

                lineage = model.distillation
                if spec.fixture_tenant_id is not None or manifest.distillation != lineage:
                    raise GatewayError(409, "distillation_lineage_mismatch")
                teacher = teacher_deployment(self, lineage.teacher_deployment_id, identity)
                if (
                    teacher.version != lineage.teacher_version
                    or teacher_digest(teacher) != lineage.teacher_artifact_digest
                    or spec.judge_version != lineage.judge_version
                    or spec.rubric_version != lineage.rubric_version
                ):
                    raise GatewayError(409, "distillation_teacher_changed")
                teacher_results = asyncio.run(
                    self._run(
                        spec.model_copy(update={"suites": ["held_out"]}),
                        {"held_out": held_out},
                        self._runner(lineage.teacher_deployment_id, identity, scratch),
                    )
                )
                teacher_scores = teacher_results[0].scores
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
        if teacher_scores is not None:
            segment_comparisons["distillation"] = self._compare(
                candidate_scores, teacher_scores, spec
            )
            for key in segments:
                segment_comparisons[f"distillation.{key}"] = self._compare(
                    [s for s in candidate_scores if key in s.segments],
                    [s for s in teacher_scores if key in s.segments],
                    spec,
                )
        report = EvaluationReport(
            specification=spec,
            candidate_manifest_version=candidate_version,
            candidate_model_version=candidate_deployment.manifest.model_version,
            candidate_artifact_digest=candidate_deployment.artifact_digest,
            baseline_manifest_version=baseline_version,
            baseline_report_id=locked.specification.evaluation_id if locked else None,
            dataset_content_digest=dataset_digest,
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
                "Public labelled fixture baseline; no held-out specialist or promotion evidence.",
                "Golden score is the fraction of exact/normalised fields matched, on a 0..1 scale.",
                "Field disagreements measured; schema and critical safety failures block locking.",
                "Replay measures recorded behavior, not current provider availability or latency.",
                "Thirty golden items and eight synthetic attacks do not establish broad safety.",
                "Synthetic planning variance retained; no real-workload variance estimate yet.",
            ]
            if spec.fixture_tenant_id is not None
            else [
                "Named golden/safety fixtures; other suites use the dataset's held-out rows.",
                "Golden measures field agreement; held-out quality uses F1/citations.",
                "Judge is not human calibrated; synthetic planning variance retained.",
                "Latency includes private persistence; no TTFT or process-memory attribution.",
                "Provider token costs exclude retrieval and infrastructure costs.",
                "No production promotion or owner approval follows from this report alone.",
            ]
            if spec.fixture_set is not None
            else [
                "Local synthetic baseline; no production promotion or owner approval.",
                "Rule-based judge is not human calibrated; human review is out of scope.",
                "Paired quality is min(F1, citation precision, recall).",
                "Synthetic pilot deltas [-0.01, 0, 0.01], sample SD 0.01; not real workload.",
                "Held-out safety segments use risk labels; adversarial tests gate safety too.",
                "No streaming TTFT, cold-start or process-memory attribution; local latency only.",
                "Costs cover fake provider tokens; retrieval and infrastructure costs unavailable.",
                "Specialist cost reduction targets do not apply to synthetic foundation tests.",
                "External artifact lifecycle remains future work.",
                "Retrieval uses immutable source snapshots and ACL decoys; no live index.",
                "Latency includes private in-memory persistence; no tenant writes or case events.",
            ],
            distillation=model.distillation if model is not None else None,
        )
        gates = decisions(report)
        report = report.model_copy(
            update={"gate_decisions": gates, "passed": all(g.passed for g in gates)}
        )
        self.store.publish(
            report,
            tenant_ids,
            identity,
            lock_baseline=locking and report.passed,
            expected_baseline=locked.specification.evaluation_id if locked else None,
            note=request.operator_note,
        )
        return self.store.get(spec.evaluation_id, identity) or report

    def _evaluate_router(
        self,
        request: EvaluationInput,
        spec: EvaluationSpecification,
        identity: Identity,
    ) -> EvaluationReport:
        from adaptive_llm.registry.artifacts import verify_artifact
        from adaptive_llm.routing.model import LogisticRouter, routing_suite

        if (
            spec.suites != ["routing"]
            or spec.baseline_deployment_id is not None
            or request.replace
            or self.registry is None
            or self.data_dir is None
        ):
            raise GatewayError(422, "routing_evaluation_configuration_invalid")
        model = self.registry.get(spec.candidate_deployment_id, identity)
        manifest, held_out = self.reader.read_routing(spec, identity)
        if (
            model.adapter_architecture != "router-logistic-v1"
            or manifest.purpose != "router_training"
            or len(model.datasets) != 1
            or model.datasets[0].version != manifest.version
            or model.datasets[0].content_digest != manifest.content_digest
            or model.datasets[0].dataset_id != manifest.dataset_id
        ):
            raise GatewayError(409, "model_dataset_mismatch")
        files = verify_artifact(
            model, self.data_dir / model.storage_location, self.persistence.keyring
        )
        router = LogisticRouter.model_validate_json(files["router.json"])
        if model.state == "candidate":
            self.registry.promote(
                PromotionRequest(
                    model_version=model.version,
                    target_state="evaluating",
                    reason="evaluation_requested",
                ),
                identity,
            )
        elif model.state in {"revoked", "deprecated"}:
            raise GatewayError(409, "model_not_evaluating")
        start = now()
        result = routing_suite(router, held_out)
        report = EvaluationReport(
            specification=spec,
            candidate_manifest_version=model.version,
            candidate_model_version=model.version,
            candidate_artifact_digest=model.artifact_digest,
            baseline_manifest_version=None,
            dataset_content_digest=manifest.content_digest,
            suite_content_digest=hashlib.sha256(b"router-logistic-v1-routing-suite-1").hexdigest(),
            code_revision=self.revision,
            started_at=start,
            completed_at=now(),
            suite_results=[result],
            baseline_suite_results=[],
            paired_comparison=PairedComparison(sample_size=0),
            segment_comparisons={},
            coverage={"routing": result.items},
            pilot_standard_deviation=0,
            derived_minimum_sample_size=4,
            gate_decisions=[],
            passed=False,
            known_limitations=[
                "Numeric counterfactual quality proxies; missing coverage labels abstention.",
                "Independent test fold; calibration uses only the validation fold.",
                "OOD detection uses unseen categories; zero count means unmeasured.",
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
            lock_baseline=False,
            expected_baseline=None,
            note=None,
        )
        return self.store.get(spec.evaluation_id, identity) or report

    @staticmethod
    def _scores(results: list[SuiteResult]) -> list[ItemScore]:
        held_out = next((result.scores for result in results if result.suite == "held_out"), None)
        return (
            held_out
            if held_out is not None
            else next((result.scores for result in results if result.suite == "golden"), [])
        )

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
