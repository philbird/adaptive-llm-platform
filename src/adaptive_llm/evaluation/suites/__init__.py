"""Replaceable suites operating on the same in-process serving Runner."""

from typing import Protocol

from adaptive_llm.contracts import EvaluationSpecification, SuiteResult
from adaptive_llm.evaluation.runner import Case, Runner


class Suite(Protocol):
    async def run(
        self, runner: Runner, cases: list[Case], specification: EvaluationSpecification
    ) -> SuiteResult: ...
