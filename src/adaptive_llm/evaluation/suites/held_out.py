"""Per-item target F1 and provenance citation checks on immutable test examples."""

from collections import Counter
from statistics import fmean

from adaptive_llm.contracts import (
    EvaluationSpecification,
    ItemScore,
    RoutingObservation,
    SuiteResult,
)
from adaptive_llm.evaluation.runner import Case, Runner
from adaptive_llm.evaluation.suites.scoring import citations, segments, token_f1


class HeldOutSuite:
    async def run(
        self, runner: Runner, cases: list[Case], specification: EvaluationSpecification
    ) -> SuiteResult:
        scores: list[ItemScore] = []
        f1s: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        failures: Counter[str] = Counter()
        for case in cases:
            outcome = await runner.run(case)
            f1 = token_f1(outcome.response.content, case.target) if outcome.response else 0.0
            precision, recall = citations(case, outcome)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            if outcome.error:
                failures["inference_error"] += 1
            if precision < 1 or recall < 1:
                failures["citation_failure"] += 1
            scores.append(
                ItemScore(
                    item_id=case.item_id,
                    score=min(f1, precision, recall),
                    segments=list(case.segments),
                    observation=RoutingObservation(
                        quality=min(f1, precision, recall),
                        validation_pass=not outcome.error,
                        cost_micros=outcome.cost_micros,
                        latency_ms=outcome.latency_ms,
                    ),
                )
            )
        return SuiteResult(
            suite="held_out",
            items=len(cases),
            metrics={
                "token_f1": fmean(f1s) if f1s else 0.0,
                "citation_precision": fmean(precisions) if precisions else 0.0,
                "citation_recall": fmean(recalls) if recalls else 0.0,
            },
            scores=scores,
            per_segment_metrics=segments(scores),
            failures=dict(failures),
        )
