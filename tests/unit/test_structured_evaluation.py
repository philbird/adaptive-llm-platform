import json
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_llm.contracts import InferenceResponse, Usage, uid
from adaptive_llm.datasets.curation import golden_texts
from adaptive_llm.evaluation.data import fixture_cases, fixture_digest, prompt_path
from adaptive_llm.evaluation.judge import DeterministicJudge, JudgeInput, structured_matches
from adaptive_llm.evaluation.runner import Outcome
from adaptive_llm.evaluation.suites.scoring import assertions


def test_dataset_decontamination_includes_structured_golden_items():
    directory = Path("tests/fixtures/golden")
    assert len(golden_texts(directory)) == 22
    assert len(golden_texts(directory, "manyfails-triage")) == 30


def test_structured_rubric_fraction_types_and_normalized_product():
    answer = JudgeInput(
        '{"is_failure":true,"kind":"incident","product":"Cedar Notes Pro"}',
        (),
        (),
        True,
        (("is_failure", True), ("kind", "product_shutdown")),
        (("product", "Ｃｅｄａｒ Notes"),),
    )
    assert structured_matches(answer) == [True, False, True]
    assert DeterministicJudge().score(answer) == pytest.approx(2 / 3)
    assert structured_matches(
        replace(answer, answer='{"is_failure":1,"kind":"incident","product":""}')
    ) == [False, False, False]
    null = JudgeInput('{"product":null}', (), (), True, (), (("product", None),))
    assert structured_matches(null) == [True]
    assert structured_matches(replace(null, answer="{}")) == [False]
    assert structured_matches(replace(answer, answer="NaN")) == [False] * 3
    assert structured_matches(replace(answer, answer="[]")) == [False] * 3


def test_fixture_prompt_path_digest_and_no_target_leakage(tmp_path):
    root = tmp_path / "fixtures"
    (root / "prompts").mkdir(parents=True)
    (root / "golden").mkdir()
    prompt = root / "prompts/policy.md"
    prompt.write_text("SYNTHETIC system policy")
    path = root / "golden/task.jsonl"
    path.write_text(
        json.dumps(
            {
                "system": "prompts/policy.md",
                "input": "SYNTHETIC input",
                "target": "SYNTHETIC private target",
                "expect_citation": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "triage", "schema": {}},
                },
            }
        )
        + "\n"
    )
    case = fixture_cases(path, "manyfails", "research-sweep")[0]
    assert case.request.messages[0].role == "system"
    assert case.request.messages[0].content == prompt.read_text()
    assert case.request.application_id == "research-sweep"
    assert case.request.max_output_tokens == 400
    assert not case.request.rag.enabled and not case.corpus
    before = fixture_digest([path])
    prompt.write_text("SYNTHETIC changed system policy")
    assert before != fixture_digest([path])
    with pytest.raises(ValueError, match="^invalid_fixture_prompt_reference$"):
        prompt_path(path, "../private.md")


def test_safety_fixture_detects_page_instructions_as_field_failures():
    cases = fixture_cases(
        Path("tests/fixtures/safety/manyfails-triage.jsonl"), "manyfails", "research-sweep"
    )
    assert len(cases) >= 8
    for case in cases:
        hit = json.loads(case.request.messages[-1].content)
        assert len(hit["page"]) <= 1500 and "SYNTHETIC" in hit["title"]
        response = InferenceResponse(
            interaction_id=uid(),
            trace_id=uid(),
            model_deployment_id="synthetic",
            content=case.target,
            citations=[],
            finish_reason="stop",
            estimated_cost_micros=0,
            usage=Usage(
                input_tokens=1, output_tokens=1, source="locally_estimated", tokenizer="synthetic"
            ),
        )
        assert all(assertions(case, Outcome(response, None, 0)).values())
        verdict = json.loads(case.target)
        verdict["is_failure"] = True
        wrong = response.model_copy(update={"content": json.dumps(verdict)})
        assert not assertions(case, Outcome(wrong, None, 0))["json_fields_mismatch"]
        assert case.critical and case.category == "injection"
