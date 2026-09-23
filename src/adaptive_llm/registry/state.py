"""Pure lifecycle rules; rollback is a separately authorised recovery operation."""

from adaptive_llm.contracts import LifecycleState
from adaptive_llm.gateway.identity import GatewayError

ALLOWED_EDGES: frozenset[tuple[LifecycleState, LifecycleState]] = frozenset(
    {
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
)


def validate_transition(
    current: LifecycleState,
    target: LifecycleState,
    *,
    actor: str | None,
    reason: str,
    evaluation_passed: bool = False,
    rollback: bool = False,
) -> None:
    allowed = (current, target) in ALLOWED_EDGES
    if rollback and (current, target) == ("deprecated", "production"):
        allowed = evaluation_passed
    if not allowed:
        raise GatewayError(409, "invalid_lifecycle_transition")
    if not actor or not reason.strip():
        raise GatewayError(422, "operator_note_required")
    if target == "approved" and not evaluation_passed:
        raise GatewayError(409, "passed_evaluation_required")
