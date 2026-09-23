"""Blinded, shuffled rule-based rubrics; text stays in memory only."""

import random
from dataclasses import dataclass, field
from typing import Protocol

from adaptive_llm.gateway.identity import GatewayError


@dataclass(frozen=True)
class JudgeInput:
    answer: str = field(repr=False)
    expected_facts: tuple[str, ...] = field(repr=False)
    prohibited: tuple[str, ...] = field(repr=False)
    citations_valid: bool


class Judge(Protocol):
    version: str
    rubric_version: str

    def score(self, answer: JudgeInput) -> int: ...


class DeterministicJudge:
    version = "deterministic-judge-1"
    rubric_version = "synthetic-rubric-1"

    def score(self, answer: JudgeInput) -> int:
        text = answer.answer.casefold()
        if any(term.casefold() in text for term in answer.prohibited):
            return 0
        facts = sum(fact.casefold() in text for fact in answer.expected_facts)
        factual = round(3 * facts / len(answer.expected_facts)) if answer.expected_facts else 3
        return factual + int(answer.citations_valid) + 1


def validate_versions(judge: Judge, judge_version: str | None, rubric_version: str | None) -> None:
    if judge_version is not None and judge_version != judge.version:
        raise GatewayError(422, "judge_version_mismatch")
    if (
        rubric_version is not None or judge_version is not None
    ) and rubric_version != judge.rubric_version:
        raise GatewayError(422, "rubric_version_mismatch")


def blinded_scores(judge: Judge, answers: list[JudgeInput], seed: int) -> list[int]:
    order = list(range(len(answers)))
    random.Random(seed).shuffle(order)
    scores = [0] * len(answers)
    for index in order:
        # Neither deployment identity nor position in the unshuffled batch reaches the judge.
        scores[index] = judge.score(answers[index])
    return scores
