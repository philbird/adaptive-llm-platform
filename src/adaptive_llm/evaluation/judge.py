"""Blinded, shuffled rule-based rubrics; text stays in memory only."""

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue

from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.structured import reject_constant


@dataclass(frozen=True)
class JudgeInput:
    answer: str = field(repr=False)
    expected_facts: tuple[str, ...] = field(repr=False)
    prohibited: tuple[str, ...] = field(repr=False)
    citations_valid: bool
    expect_json_fields: tuple[tuple[str, JsonValue], ...] = field(default=(), repr=False)
    expect_json_text_match: tuple[tuple[str, str | None], ...] = field(default=(), repr=False)


def structured_matches(answer: JudgeInput) -> list[bool]:
    try:
        obj = json.loads(answer.answer, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        obj = None
    if not isinstance(obj, dict):
        return [False] * (len(answer.expect_json_fields) + len(answer.expect_json_text_match))

    def normalize(text: str) -> str:
        return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))

    matches = [
        key in obj and type(obj[key]) is type(value) and obj[key] == value
        for key, value in answer.expect_json_fields
    ]
    for key, expected in answer.expect_json_text_match:
        actual = obj.get(key)
        if expected is None:
            matches.append(key in obj and actual is None)
        elif isinstance(actual, str):
            left, right = normalize(actual), normalize(expected)
            matches.append(bool(left and right) and (left in right or right in left))
        else:
            matches.append(False)
    return matches


class Judge(Protocol):
    version: str
    rubric_version: str

    def score(self, answer: JudgeInput) -> float: ...


class DeterministicJudge:
    version = "deterministic-judge-1"
    rubric_version = "synthetic-rubric-1"

    def score(self, answer: JudgeInput) -> float:
        text = answer.answer.casefold()
        if any(term.casefold() in text for term in answer.prohibited):
            return 0
        matches = structured_matches(answer)
        if matches:
            return sum(matches) / len(matches)
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


def blinded_scores(judge: Judge, answers: list[JudgeInput], seed: int) -> list[float]:
    order = list(range(len(answers)))
    random.Random(seed).shuffle(order)
    scores = [0.0] * len(answers)
    for index in order:
        # Neither deployment identity nor position in the unshuffled batch reaches the judge.
        scores[index] = judge.score(answers[index])
    return scores
