from time import monotonic, sleep
from typing import TYPE_CHECKING

import pytest

from adaptive_llm.contracts import RoutePolicy, uid
from adaptive_llm.routing.live import assigned

if TYPE_CHECKING:
    from conftest import RouterSeed

OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}
USER = {"Authorization": "Bearer synthetic-key-a"}
NOTE = {"reason": "SYNTHETIC live routing"}


def infer(
    seed: "RouterSeed", *, mode: str = "auto", content: str = "SYNTHETIC live example"
) -> dict:
    result = seed.seed.client.post(
        "/v1/inference",
        headers=USER,
        json={
            "request_id": uid(),
            "application_id": "support-assistant",
            "max_output_tokens": 32,
            "messages": [{"role": "user", "content": content}],
            "routing": {"mode": mode},
        },
    )
    assert result.status_code == 200, result.json()
    return result.json()


def promote(seed: "RouterSeed", target: str, expected: int = 200) -> dict:
    result = seed.seed.client.post(
        f"/v1/models/{seed.specialist}/promotion-requests",
        headers=OPERATOR,
        json={"model_version": seed.specialist, "target_state": target, **NOTE},
    )
    assert result.status_code == expected, result.json()
    return result.json()


def policy(seed: "RouterSeed", **updates: object) -> RoutePolicy:
    value = RoutePolicy.model_validate(
        {
            "eligible_specialist_versions": [seed.specialist],
            "shadow_enabled": True,
            "live_specialists_allowed": True,
            "router_version": seed.router,
            "canary": {"traffic_fraction": 0.05},
            "tenant_enabled": {"synthetic-a": True},
            "task_enabled": {"general": True},
            **updates,
        }
    )
    result = seed.seed.client.post(
        "/v1/route-policies", headers=OPERATOR, json=value.model_dump(mode="json")
    )
    assert result.status_code == 200, result.json()
    result = seed.seed.client.post(
        f"/v1/route-policies/{value.policy_id}/activate", headers=OPERATOR, json=NOTE
    )
    assert result.status_code == 200, result.json()
    return value


def prepare_canary(
    seed: "RouterSeed",
    monkeypatch: pytest.MonkeyPatch,
    **updates: object,
) -> RoutePolicy:
    p = policy(seed, **updates)
    assert promote(seed, "canary", 409)["error"]["code"] == "passed_shadow_report_required"
    for _ in range(4):
        result = infer(seed, mode="specialist")
        assert not result["route"]["specialist_served"]  # shadow state cannot serve
    assert seed.seed.client.portal is not None
    seed.seed.client.portal.call(seed.seed.app.state.shadow.queue.join)
    promote(seed, "canary")
    # Admission manifests follow the shared one-second control snapshot cadence.
    snapshot = seed.seed.app.state.route_policies.snapshot()
    deadline = monotonic() + 2
    while seed.seed.app.state.route_policies.snapshot() is snapshot:
        assert monotonic() < deadline
        sleep(0.01)

    def chosen() -> str:
        while True:
            value = uid()
            if assigned(value, p):
                return value

    monkeypatch.setattr("adaptive_llm.gateway.service.uid", chosen)
    return p
