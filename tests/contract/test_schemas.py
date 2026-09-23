import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm import contracts
from adaptive_llm.app import create_app
from adaptive_llm.contracts import Event, Started

ROOT = Path(__file__).resolve().parents[2]


def test_generated_schemas_match_models() -> None:
    for name, cls in vars(contracts).items():
        if (
            isinstance(cls, type)
            and issubclass(cls, contracts.Contract)
            and cls is not contracts.Contract
        ):
            folder = "events" if cls is Event else "schemas"
            assert (
                json.loads((ROOT / "contracts" / folder / f"{name}.json").read_text())
                == cls.model_json_schema()
            )
    assert (
        json.loads((ROOT / "contracts/openapi/openapi.json").read_text()) == create_app().openapi()
    )


def test_event_roundtrip_and_type_mismatch() -> None:
    data = Started(interaction_id="synthetic", application_id="demo", policy_version="local-1")
    event = Event(
        event_type="interaction.started.v1", tenant_id="demo", trace_id="trace", data=data
    )
    assert Event.model_validate_json(event.model_dump_json()) == event
    with pytest.raises(ValidationError):
        Event(event_type="retrieval.completed.v1", tenant_id="demo", trace_id="trace", data=data)


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
