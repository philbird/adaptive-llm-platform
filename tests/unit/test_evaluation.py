import json
import math
from dataclasses import fields
from pathlib import Path
from statistics import NormalDist

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import (
    EvaluationInput,
    EvaluationSpecification,
    PairedComparison,
    SuiteResult,
    uid,
)
from adaptive_llm.evaluation.judge import DeterministicJudge, JudgeInput, blinded_scores
from adaptive_llm.evaluation.runner import EvaluationPolicy
from adaptive_llm.evaluation.stats import paired_bootstrap, percentile, required_sample_size
from adaptive_llm.evaluation.suites.scoring import token_f1
from adaptive_llm.gateway.identity import Identity


def test_paired_bootstrap_known_values_and_repeatability() -> None:
    result = paired_bootstrap([1, 0.75, 0.5], [0.5, 0.25, 0])
    assert result == PairedComparison(mean_delta=0.5, ci_lower=0.5, ci_upper=0.5, sample_size=3)
    assert paired_bootstrap([0, 0.5, 1], [0.5] * 3, seed=7) == paired_bootstrap(
        [0, 0.5, 1], [0.5] * 3, seed=7
    )
    assert paired_bootstrap([], []).sample_size == 0
    with pytest.raises(ValueError, match="unpaired_samples"):
        paired_bootstrap([1], [])
    with pytest.raises(ValueError, match="nonfinite_samples"):
        paired_bootstrap([math.nan], [0])
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_sample_size_strict_half_width_and_pinned_configuration() -> None:
    n = required_sample_size(0.02, 0.01, 0.95)
    assert n == 4
    z = NormalDist().inv_cdf(0.975)
    assert z * 0.01 / math.sqrt(n) < 0.01
    assert z * 0.01 / math.sqrt(n - 1) >= 0.01
    config = json.loads(Path("configs/evaluation/initial-targets.json").read_text())
    assert config["minimum_sample_sizes"]["overall"] == n
    assert config["minimum_sample_sizes"]["per_critical_segment"] == n
    assert required_sample_size(0.02, 0, 0.95) == 1
    for margin, sd, confidence in [(0, 1, 0.95), (0.02, -1, 0.95), (0.02, 1, 1)]:
        with pytest.raises(ValueError):
            required_sample_size(margin, sd, confidence)


def test_judge_is_blinded_shuffled_pinned_and_scores_zero_to_five() -> None:
    answers = [
        JudgeInput(f"SYNTHETIC fact {i}", (f"fact {i}",), ("prohibited",), True) for i in range(8)
    ]
    seen = []

    class SpyJudge(DeterministicJudge):
        def score(self, answer: JudgeInput) -> int:
            seen.append(answer)
            return super().score(answer)

    judge = SpyJudge()
    assert blinded_scores(judge, answers, 23) == [5] * 8
    assert seen != answers and set(seen) == set(answers)
    assert all("model" not in f.name and "deployment" not in f.name for f in fields(JudgeInput))
    assert judge.version == "deterministic-judge-1"
    assert judge.rubric_version == "synthetic-rubric-1"
    assert judge.score(JudgeInput("SYNTHETIC prohibited", (), ("prohibited",), True)) == 0
    assert judge.score(JudgeInput("SYNTHETIC", ("absent",), (), False)) == 1


def test_f1_normalisation_and_multiplicity() -> None:
    assert token_f1("ＡＢＣ, abc!", "abc") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1
    assert token_f1("", "synthetic") == 0
    assert token_f1("synthetic", "different") == 0


def test_contracts_bounds_additive_records_and_evaluation_policy() -> None:
    spec = dict(
        candidate_deployment_id="fake",
        baseline_deployment_id=None,
        dataset_id="synthetic",
        dataset_version=uid(),
        suites=["golden"],
    )
    assert EvaluationSpecification(**spec, future=True).minimum_sample_size is None
    for update in [
        {"evaluation_id": "../unsafe"},
        {"dataset_version": ".."},
        {"suites": ["golden", "golden"]},
    ]:
        with pytest.raises(ValidationError):
            EvaluationSpecification(**(spec | update))
    with pytest.raises(ValidationError):
        EvaluationInput(**spec, replace=True)
    with pytest.raises(ValidationError):
        EvaluationInput(**spec, future=True)
    with pytest.raises(ValidationError):
        SuiteResult(suite="golden", items=1, metrics={str(i): 0.0 for i in range(41)})
    identity = Identity("synthetic-a", frozenset({"evaluation"}), "local", None)
    policy = EvaluationPolicy().decide(identity, "evaluation")
    assert (
        policy.processing_allowed
        and not policy.training_allowed
        and not policy.content_logging_allowed
    )
    assert not EvaluationPolicy().decide(identity, "support-assistant").processing_allowed


@pytest.mark.asyncio
async def test_suites_measure_bad_rank_errors_disagreements_and_failed_outcome_costs() -> None:
    from dataclasses import replace

    from adaptive_llm.contracts import (
        ChunkEvidence,
        Citation,
        InferenceResponse,
        RetrievalRun,
        Usage,
    )
    from adaptive_llm.evaluation.data import fixture_cases
    from adaptive_llm.evaluation.runner import Case, Outcome
    from adaptive_llm.evaluation.suites.golden import GoldenSuite
    from adaptive_llm.evaluation.suites.performance import PerformanceSuite
    from adaptive_llm.evaluation.suites.retrieval import RetrievalSuite
    from adaptive_llm.rag import RetrievalResult

    case = fixture_cases(Path("tests/fixtures/golden/synthetic.jsonl"), "synthetic-a")[1]
    spec = EvaluationSpecification(
        candidate_deployment_id="fake",
        baseline_deployment_id=None,
        dataset_id="synthetic",
        dataset_version=uid(),
        suites=["golden"],
        performance_requests=2,
        concurrency=1,
    )
    relevant = case.corpus[0]
    distractor = relevant.model_copy(update={"document_id": "synthetic-distractor"})
    candidates = [
        ChunkEvidence(
            document_id=c.document_id,
            document_version=c.document_version,
            chunk_id=c.chunk_id,
            rank_retrieved=i + 1,
            retrieval_score=1,
            supplied_to_model=True,
            context_position=i,
            token_count=1,
            content_hash="synthetic",
            licence_class="synthetic",
        )
        for i, c in enumerate([distractor, relevant])
    ]
    retrieval = RetrievalResult(
        RetrievalRun(
            interaction_id=uid(),
            index_id=relevant.index_id,
            index_version=relevant.index_version,
            query_hash=None,
            latency_ms=0,
            candidates=candidates,
        ),
        (distractor, relevant),
    )
    response = InferenceResponse(
        interaction_id=uid(),
        trace_id=uid(),
        model_deployment_id="fake",
        content=case.target,
        citations=[Citation(document_id=relevant.document_id, chunk_id=relevant.chunk_id)],
        usage=Usage(input_tokens=1, output_tokens=1, source="locally_estimated", tokenizer="fake"),
        estimated_cost_micros=7,
        finish_reason="stop",
    )

    class StubRunner:
        calls = 0
        fail_alternate = False

        async def run(self, case: Case) -> Outcome:
            self.calls += 1
            failed = self.fail_alternate and self.calls % 2 == 0
            return Outcome(None if failed else response, retrieval, 10 * self.calls, failed, 7)

    runner = StubRunner()
    result = await RetrievalSuite().run(runner, [case], spec)
    assert result.metrics["recall_at_k"] == 1
    assert result.metrics["mrr"] == 0.5
    assert result.metrics["context_precision"] == 0.5
    result = await RetrievalSuite().run(runner, [replace(case, k=1)], spec)
    assert result.metrics["recall_at_k"] == result.metrics["mrr"] == 0
    assert result.failures["retrieval_failure"] == 1

    class DisagreeingJudge(DeterministicJudge):
        def score(self, answer: JudgeInput) -> int:
            return 0

    result = await GoldenSuite(DisagreeingJudge()).run(runner, [case], spec)
    assert result.metrics["assertion_pass_rate"] == 1
    assert result.metrics["judge_disagreements"] == 1
    runner.calls = 0
    runner.fail_alternate = True
    result = await PerformanceSuite().run(runner, [case], spec)
    assert result.metrics["error_rate"] == 0.5
    assert result.metrics["p50_latency_ms"] == 15
    assert result.total_cost_micros == result.cost_per_success_micros == 14
