import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm import contracts
from adaptive_llm.app import create_app
from adaptive_llm.contracts import EVENT_PAYLOADS, DeletionRequest, Event, Started

ROOT = Path(__file__).resolve().parents[2]


def contract_classes() -> dict[str, type[contracts.Contract]]:
    return {
        name: cls
        for name, cls in vars(contracts).items()
        if isinstance(cls, type)
        and issubclass(cls, contracts.Contract)
        and cls not in (contracts.Contract, contracts.Record)
    }


def test_generated_schemas_match_models() -> None:
    for name, cls in contract_classes().items():
        folder = "events" if cls is Event else "schemas"
        assert (
            json.loads((ROOT / "contracts" / folder / f"{name}.json").read_text())
            == cls.model_json_schema()
        ), name
    assert (
        json.loads((ROOT / "contracts/openapi/openapi.json").read_text()) == create_app().openapi()
    )


def test_no_stale_schema_files() -> None:
    expected = {f"{name}.json" for name, cls in contract_classes().items() if cls is not Event}
    assert {p.name for p in (ROOT / "contracts/schemas").glob("*.json")} == expected


def test_event_roundtrip_and_type_mismatch() -> None:
    data = Started(interaction_id="synthetic", application_id="demo", policy_version="local-1")
    event = Event(
        event_type="interaction.started.v1", tenant_id="demo", trace_id="trace", data=data
    )
    assert Event.model_validate_json(event.model_dump_json()) == event
    with pytest.raises(ValidationError):
        Event(event_type="retrieval.completed.v1", tenant_id="demo", trace_id="trace", data=data)


def test_every_spec_event_type_has_a_payload() -> None:
    spec_types = {
        "interaction.started.v1",
        "retrieval.completed.v1",
        "route.decided.v1",
        "generation.completed.v1",
        "generation.failed.v1",
        "interaction.completed.v1",
        "feedback.recorded.v1",
        "privacy.deletion.requested.v1",
        "dataset.built.v1",
        "training.completed.v1",
        "evaluation.completed.v1",
        "deployment.changed.v1",
    }
    assert set(EVENT_PAYLOADS) == spec_types
    deletion = Event(
        event_type="privacy.deletion.requested.v1",
        producer="policy_redaction",
        tenant_id="demo",
        trace_id="trace",
        data=DeletionRequest(scope="subject", target_id="hmac-ref"),
    )
    assert Event.model_validate_json(deletion.model_dump_json()) == deletion


def test_unknown_schema_versions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Started.model_validate(
            {
                "schema_version": "2.0",
                "interaction_id": "synthetic",
                "application_id": "demo",
                "policy_version": "local-1",
            }
        )
