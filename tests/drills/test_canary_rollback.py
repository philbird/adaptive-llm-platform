import json
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from routing_helpers import USER, infer, policy, prepare_canary

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Interaction, LiveObservation, RouteDecision, RoutingFeatures, uid


@pytest.mark.drill
def test_rollback_breach_to_foundation_under_five_seconds(router_seed, monkeypatch) -> None:
    r = router_seed
    p = prepare_canary(r, monkeypatch)
    other = create_app(Settings(data_dir=r.seed.directory, outbox_dispatch_enabled=False))
    with TestClient(other) as client:

        def request():
            result = client.post(
                "/v1/inference",
                headers=USER,
                json={
                    "request_id": uid(),
                    "application_id": "support-assistant",
                    "max_output_tokens": 32,
                    "messages": [{"role": "user", "content": "SYNTHETIC rollback drill"}],
                },
            )
            assert result.status_code == 200
            return result.json()

        assert request()["route"]["specialist_served"]
        started = monotonic()
        r.seed.app.state.route_policies.record_live(
            LiveObservation(
                interaction_id=uid(),
                policy_id=p.policy_id,
                tenant_id="synthetic-a",
                features=RoutingFeatures(task="general", input_tokens=4),
                specialist_version=r.specialist,
                specialist_served=True,
                success=True,
                critical_safety_incidents=1,
                quality=0,
                total_cost_micros=1,
                latency_ms=1,
            )
        )
        while r.specialist not in other.state.route_policies.snapshot().disabled_specialists:
            assert monotonic() - started < 5
            sleep(0.02)
        response = request()
        elapsed = monotonic() - started
        assert not response["route"]["specialist_served"] and elapsed < 5
        records = r.seed.app.state.evaluation_database.connection.execute(
            "SELECT data FROM rollback_measurements WHERE specialist_version=?",
            (r.specialist,),
        ).fetchall()
        assert len(records) == 1
        assert "critical_safety_incident" in json.loads(records[0][0])["breach_reasons"]
        events = r.seed.app.state.evaluation_database.connection.execute(
            "SELECT envelope FROM outbox WHERE event_type='deployment.changed.v1' "
            "AND json_extract(envelope, '$.data.model_version')=? "
            "AND json_extract(envelope, '$.data.new_state')='disabled'",
            (r.specialist,),
        ).fetchall()
        assert events and all(
            "critical_safety_incident" in json.loads(event[0])["data"]["reason"] for event in events
        )
        print(
            f"automatic-rollback: breach_to_foundation_seconds={elapsed:.3f}; "
            "persistent_disabled=true; second_instance=honoured"
        )


@pytest.mark.drill
def test_novel_task_goes_to_foundation(router_seed, monkeypatch) -> None:
    r = router_seed
    prepare_canary(r, monkeypatch)
    # Retain canary state; use an active policy enabling a task absent from the router's fit fold.
    p = policy(r, task_enabled={"general": True, "question_answering": True})
    from adaptive_llm.routing.live import assigned

    def chosen():
        while True:
            value = uid()
            if assigned(value, p):
                return value

    monkeypatch.setattr("adaptive_llm.gateway.service.uid", chosen)
    result = r.seed.client.post(
        "/v1/inference",
        headers=USER,
        json={
            "request_id": uid(),
            "application_id": "support-assistant",
            "max_output_tokens": 32,
            "messages": [{"role": "user", "content": "SYNTHETIC novel question unused receipt"}],
            "rag": {"enabled": True, "index_id": "synthetic-kb"},
        },
    )
    assert result.status_code == 200
    assert not result.json()["route"]["specialist_served"]
    interaction = r.seed.app.state.metadata.get(
        "synthetic-a", Interaction, result.json()["interaction_id"]
    )
    route = r.seed.app.state.metadata.get(
        "synthetic-a", RouteDecision, interaction.route_decision_id
    )
    assert "out_of_distribution" in route.fallback_reasons
    assert route.candidates[0].ood_score > p.ood_threshold_max
    assert infer(r)["route"]["specialist_served"]
    print(
        f"novel-task: foundation=true; reason=out_of_distribution; "
        f"ood={route.candidates[0].ood_score:.3f}; max={p.ood_threshold_max}"
    )
