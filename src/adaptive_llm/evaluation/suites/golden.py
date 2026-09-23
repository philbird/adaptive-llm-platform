"""Golden assertions plus an independent, pinned and blinded deterministic rubric."""

from collections import Counter
from statistics import fmean

from adaptive_llm.contracts import EvaluationSpecification, SuiteResult
from adaptive_llm.evaluation.judge import Judge, JudgeInput, blinded_scores, validate_versions
from adaptive_llm.evaluation.runner import Case, Runner
from adaptive_llm.evaluation.suites.scoring import assertions, citations


class GoldenSuite:
    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    async def run(
        self, runner: Runner, cases: list[Case], specification: EvaluationSpecification
    ) -> SuiteResult:
        validate_versions(self.judge, specification.judge_version, specification.rubric_version)
        failures: Counter[str] = Counter()
        passed: list[bool] = []
        inputs: list[JudgeInput] = []
        for case in cases:
            outcome = await runner.run(case)
            checks = assertions(case, outcome)
            failures.update(reason for reason, ok in checks.items() if not ok)
            passed.append(all(checks.values()))
            inputs.append(
                JudgeInput(
                    outcome.response.content if outcome.response else "",
                    case.expected_facts,
                    case.prohibited,
                    citations(case, outcome) == (1, 1),
                )
            )
        metrics = {"assertion_pass_rate": fmean(passed) if passed else 0.0}
        if specification.judge_version is not None:
            scores = blinded_scores(self.judge, inputs, specification.seed)
            metrics.update(
                rubric_score=fmean(scores) if scores else 0.0,
                judge_disagreements=float(
                    sum(ok != (score == 5) for ok, score in zip(passed, scores, strict=True))
                ),
            )
        return SuiteResult(
            suite="golden", items=len(cases), metrics=metrics, failures=dict(failures)
        )
