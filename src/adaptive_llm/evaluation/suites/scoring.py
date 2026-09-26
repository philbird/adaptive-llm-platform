"""Content stays in memory; only bounded numerical aggregates leave a suite."""

import json
import re
import unicodedata
from collections import Counter
from statistics import fmean

from adaptive_llm.contracts import ItemScore
from adaptive_llm.evaluation.judge import JudgeInput, structured_matches
from adaptive_llm.evaluation.runner import Case, Outcome


def token_f1(answer: str, target: str) -> float:
    def tokens(text: str) -> Counter[str]:
        return Counter(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))

    actual, expected = tokens(answer), tokens(target)
    if not actual or not expected:
        return float(actual == expected)
    overlap = sum((actual & expected).values())
    return 2 * overlap / (sum(actual.values()) + sum(expected.values()))


def citations(case: Case, outcome: Outcome) -> tuple[float, float]:
    if outcome.response is None:
        return 0.0, 0.0
    actual = {(c.document_id, c.chunk_id) for c in outcome.response.citations}
    expected = case.expected_citations
    correct = len(actual & expected)
    precision = correct / len(actual) if actual else float(not expected)
    recall = correct / len(expected) if expected else 1.0
    return precision, recall


def assertions(case: Case, outcome: Outcome) -> dict[str, bool]:
    response = outcome.response
    text = response.content if response else ""
    precision, recall = citations(case, outcome)
    result = {
        "inference_error": response is not None and response.finish_reason == "stop",
        "expected_fact_missing": all(f.casefold() in text.casefold() for f in case.expected_facts),
        "citation_failure": precision == recall == 1,
        "prohibited_content": not any(
            word.casefold() in text.casefold() for word in case.prohibited
        ),
    }
    if case.expect_json:
        try:
            result["invalid_json"] = isinstance(json.loads(text), dict)
        except ValueError:
            result["invalid_json"] = False
    if case.expect_json_fields or case.expect_json_text_match:
        matches = structured_matches(
            JudgeInput(
                text,
                (),
                (),
                True,
                tuple(case.expect_json_fields.items()),
                tuple(case.expect_json_text_match.items()),
            )
        )
        result["json_fields_mismatch"] = all(matches)
    return result


def segments(scores: list[ItemScore]) -> dict[str, dict[str, float]]:
    keys = sorted({key for score in scores for key in score.segments})
    return {
        key: {
            "score": fmean(score.score for score in scores if key in score.segments),
            "n": float(sum(key in score.segments for score in scores)),
        }
        for key in keys
    }
