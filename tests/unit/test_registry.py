from itertools import product
from typing import get_args

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import LifecycleState, OperatorNote, TrainingJobSpecification, uid
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.registry.state import validate_transition


def test_every_pair_of_lifecycle_states() -> None:
    expected = {
        ("candidate", "evaluating"),
        ("evaluating", "approved"),
        ("approved", "shadow"),
        ("shadow", "canary"),
        ("canary", "production"),
        ("production", "deprecated"),
        ("candidate", "revoked"),
        ("evaluating", "revoked"),
        ("approved", "revoked"),
        ("shadow", "revoked"),
        ("canary", "revoked"),
        ("production", "revoked"),
        ("deprecated", "revoked"),
    }
    for current, target in product(get_args(LifecycleState), repeat=2):
        if (current, target) in expected:
            validate_transition(
                current, target, actor="operator", reason="synthetic", evaluation_passed=True
            )
        else:
            with pytest.raises(GatewayError, match="invalid_lifecycle_transition"):
                validate_transition(
                    current, target, actor="operator", reason="synthetic", evaluation_passed=True
                )


def test_gate_and_operator_preconditions_and_rollback_exception() -> None:
    with pytest.raises(GatewayError, match="passed_evaluation_required"):
        validate_transition("evaluating", "approved", actor="operator", reason="synthetic")
    for actor, reason in [(None, "synthetic"), ("operator", " ")]:
        for current, target in [
            ("approved", "shadow"),
            ("shadow", "canary"),
            ("canary", "production"),
            ("candidate", "revoked"),
        ]:
            with pytest.raises(GatewayError, match="operator_note_required"):
                validate_transition(current, target, actor=actor, reason=reason)
    validate_transition(
        "deprecated",
        "production",
        actor="operator",
        reason="synthetic",
        evaluation_passed=True,
        rollback=True,
    )
    with pytest.raises(GatewayError):
        validate_transition(
            "deprecated", "production", actor="operator", reason="synthetic", rollback=True
        )


@pytest.mark.parametrize(
    "update",
    [
        {"job_type": "sft"},
        {"job_id": "../escape"},
        {"registry_id": ".."},
        {"adapter_config": {"rank": 0}},
        {"container_digest": "latest"},
    ],
)
def test_training_contract_rejects_unsupported_and_unsafe_config(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainingJobSpecification.model_validate(
            {"dataset_id": "synthetic", "dataset_version": uid(), **update}
        )
    with pytest.raises(ValidationError):
        OperatorNote(reason=" ")
