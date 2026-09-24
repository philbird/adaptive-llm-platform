import json
from dataclasses import replace
from datetime import timedelta
from time import monotonic, sleep
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    EvaluationInput,
    GenerationAttempt,
    Interaction,
    RouteDecision,
    RoutingRow,
    SourceWindow,
    TrainingJobSpecification,
    now,
    uid,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.providers import ProviderRequest, ProviderResult

if TYPE_CHECKING:
    from conftest import RouterSeed

from routing_helpers import NOTE, OPERATOR, USER, infer, policy, prepare_canary, promote


def test_router_training_artifacts_reproducible_and_numeric_only(router_seed: "RouterSeed") -> None:
    r = router_seed
    seed = r.seed
    identity = seed.app.state.authenticator.authenticate(OPERATOR["Authorization"], None)
    model = seed.app.state.registry.get(r.router, identity)
    assert model.adapter_architecture == "router-logistic-v1"
    assert model.state == "approved"
    evaluation = seed.app.state.evaluations.get(next(iter(model.evaluation_reports)), identity)
    print("router-evaluation: " + json.dumps(evaluation.suite_results[0].metrics))
    dataset = seed.app.state.datasets.get(r.dataset.dataset_id, r.dataset.version, identity)
    shards = read_shards(
        dataset, seed.directory, seed.app.state.persistence.cipher, seed.app.state.keyring
    )
    for values in shards.values():
        for value in values:
            row = RoutingRow.model_validate_json(value)
            assert row.source_dataset_version == seed.manifest.version
            assert not any(
                key in json.loads(value) for key in ("input", "target", "messages", "text")
            )
            assert b"SYNTHETIC routing example" not in value
    from conftest import wait_training

    spec = TrainingJobSpecification(
        job_type="router",
        registry_id="synthetic-router",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
    )
    assert (
        seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
        ).status_code
        == 200
    )
    second = wait_training(seed.client, spec.job_id, OPERATOR)
    assert second.state == "succeeded"
    assert second.artifact_digest == model.artifact_digest
    # Pending datasets are never accepted for router jobs either.
    fresh = seed.app.state.datasets.build(dataset.specification, identity)
    response = seed.client.post(
        "/v1/training/jobs",
        headers=OPERATOR,
        json=spec.model_copy(update={"job_id": uid(), "dataset_version": fresh.version}).model_dump(
            mode="json"
        ),
    )
    assert response.status_code == 409
    # Held-out interactions without shadow coverage are retained as null specialist columns.
    source = dataset.specification.model_copy(
        update={
            "source_window": SourceWindow(start=seed.manifest.source_window.start, end=now()),
            "time_split": None,
        }
    )
    combined = seed.app.state.datasets.build(source, identity)
    all_rows = read_shards(
        combined, seed.directory, seed.app.state.persistence.cipher, seed.app.state.keyring
    )
    held_out = [
        RoutingRow.model_validate_json(v)
        for split in all_rows.values()
        for v in split
        if json.loads(v)["source_example_hash"] is not None
    ]
    assert len(held_out) == 8
    # Offline held-out scores take priority even with no shadow observation.
    assert all(row.candidates[r.specialist].quality == 1 for row in held_out)
    assert (
        seed.client.post("/v1/route-policies/kill-switch", headers=OPERATOR, json=NOTE).status_code
        == 200
    )
    gap = infer(r)["interaction_id"]
    combined = seed.app.state.datasets.build(
        source.model_copy(
            update={
                "source_window": SourceWindow(start=seed.manifest.source_window.start, end=now()),
            }
        ),
        identity,
    )
    rows_with_gap = read_shards(
        combined, seed.directory, seed.app.state.persistence.cipher, seed.app.state.keyring
    )
    gap_row = next(
        RoutingRow.model_validate_json(v)
        for split in rows_with_gap.values()
        for v in split
        if json.loads(v)["interaction_id"] == gap
    )
    assert gap_row.candidates[r.specialist] is None
    seed.app.state.training.policy.training["synthetic-a"] = False
    denied = seed.client.post(
        "/v1/training/jobs",
        headers=OPERATOR,
        json=spec.model_copy(update={"job_id": uid()}).model_dump(mode="json"),
    )
    assert denied.status_code == 403


def test_live_canary_progression_mode_accounting_and_restart(
    router_seed: "RouterSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = router_seed
    p = prepare_canary(r, monkeypatch)
    assert promote(r, "production", 409)["error"]["code"] == "passed_canary_report_required"
    for _ in range(4):
        result = infer(r, mode="specialist")
        assert result["model_deployment_id"] == r.specialist
        assert result["route"]["specialist_served"] and not result["route"]["fallback_used"]
        interaction = r.seed.app.state.metadata.get(
            "synthetic-a", Interaction, result["interaction_id"]
        )
        assert interaction.total_cost_micros == result["estimated_cost_micros"]
        route = r.seed.app.state.metadata.get(
            "synthetic-a", RouteDecision, interaction.route_decision_id
        )
        assert route.router_version == r.router and route.candidates[0].confidence > 0.9
    assert not infer(r, mode="foundation")["route"]["specialist_served"]
    constrained = r.seed.client.post(
        "/v1/inference",
        headers=USER,
        json={
            "request_id": uid(),
            "application_id": "support-assistant",
            "max_output_tokens": 32,
            "messages": [{"role": "user", "content": "SYNTHETIC budget request"}],
            "routing": {"mode": "specialist", "max_cost_micros": 20},
        },
    )
    assert constrained.status_code == 200 and constrained.json()["route"]["specialist_served"]
    report = r.seed.client.get(
        "/v1/canary/reports", headers=OPERATOR, params={"policy": p.policy_id, "since": p_time()}
    )
    assert report.status_code == 200, report.json()
    assert report.json()["passed"], report.json()
    assert report.json()["overall"]["specialist"]["interactions"] == 5
    assert report.json()["overall"]["cost_reduction_fraction"] >= 0.3
    print(f"live-canary-p95-ms={report.json()['overall']['specialist']['p95_latency_ms']:.3f}")
    print(
        "live-canary: "
        + json.dumps(
            {
                key: report.json()["overall"][key]
                for key in ("cost_reduction_fraction", "quality_delta")
            }
        )
    )
    promote(r, "production")
    expanded = policy(r, canary={"traffic_fraction": 1})
    assert expanded.canary.traffic_fraction == 1
    # Disablement survives a new control-store connection and cannot be cleared without a note.
    assert (
        r.seed.client.post(
            "/v1/route-policies/kill-switch",
            headers=OPERATOR,
            json={"specialist_version": r.specialist, **NOTE},
        ).status_code
        == 200
    )
    assert (
        r.seed.client.request(
            "DELETE",
            "/v1/route-policies/kill-switch",
            headers=OPERATOR,
            json={"specialist_version": r.specialist, "reason": " "},
        ).status_code
        == 422
    )
    with TestClient(
        create_app(Settings(data_dir=r.seed.directory, outbox_dispatch_enabled=False))
    ) as client:
        result = client.post(
            "/v1/inference",
            headers=USER,
            json={
                "request_id": uid(),
                "application_id": "support-assistant",
                "max_output_tokens": 32,
                "messages": [{"role": "user", "content": "SYNTHETIC restart"}],
            },
        )
        assert result.status_code == 200
        assert not result.json()["route"]["specialist_served"]
    assert (
        r.seed.client.request(
            "DELETE",
            "/v1/route-policies/kill-switch",
            headers=OPERATOR,
            json={"specialist_version": r.specialist, **NOTE},
        ).status_code
        == 200
    )
    assert infer(r)["route"]["specialist_served"]


def p_time() -> str:
    return (now() - timedelta(minutes=5)).isoformat()


def test_registry_outage_falls_back_and_is_visible_in_health(router_seed, monkeypatch):
    r = router_seed
    prepare_canary(r, monkeypatch)
    assert infer(r)["route"]["specialist_served"]
    assert r.seed.client.get("/healthz").json()["live_planner_failures"] == 0
    # A new policy snapshot forces the registry refresh; cached routers cannot hide an outage.
    policy(r)

    def unavailable(*args):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(r.seed.app.state.registry, "get", unavailable)
    result = infer(r)
    assert not result["route"]["specialist_served"]
    interaction = r.seed.app.state.metadata.get(
        "synthetic-a", Interaction, result["interaction_id"]
    )
    decision = r.seed.app.state.metadata.get(
        "synthetic-a", RouteDecision, interaction.route_decision_id
    )
    assert decision.fallback_reasons == ["policy_uncertainty"]
    health = r.seed.client.get("/healthz").json()
    assert health["live_planner_failures"] == 1
    assert not health["kill_switch"]


def test_live_failure_falls_back_and_monitor_disables(
    router_seed: "RouterSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = router_seed
    p = prepare_canary(r, monkeypatch)
    planner = r.seed.app.state.inference.chain_planner
    loader = planner.loader

    class FailingLoader:
        def load(self, version, identity):
            candidate = loader.load(version, identity)

            class FailingProvider:
                async def generate(self, request: ProviderRequest) -> ProviderResult:
                    result = await candidate.provider.generate(request)
                    return replace(result, content="")

            return replace(candidate, provider=FailingProvider())

    planner.loader = FailingLoader()
    for _ in range(4):
        result = infer(r)
        assert result["route"]["fallback_used"]
        assert not result["route"]["specialist_served"]
        assert result["content"]
        assert result["estimated_cost_micros"] > 10
    deadline = monotonic() + 5
    while r.specialist not in r.seed.app.state.route_policies.snapshot().disabled_specialists:
        assert monotonic() < deadline
        sleep(0.02)
    measurement = r.seed.app.state.evaluation_database.connection.execute(
        "SELECT data FROM rollback_measurements WHERE specialist_version=?",
        (r.specialist,),
    ).fetchone()
    assert "validation_failure_rate" in json.loads(measurement[0])["breach_reasons"]
    assert not infer(r)["route"]["specialist_served"]
    assert promote(r, "production", 409)["error"]["code"] == "deployment_disabled_or_unconfigured"
    assert (
        r.seed.client.get(
            "/v1/canary/reports", headers=USER, params={"policy": p.policy_id, "since": p_time()}
        ).status_code
        == 403
    )


def test_cost_exit_mix_includes_every_attempt(router_seed, monkeypatch):
    r = router_seed
    p = prepare_canary(r, monkeypatch, rollback={"validation_failure_rate_max": 0.25})
    foundation = [infer(r, mode="foundation") for _ in range(6)]
    foundation_unit_cost = foundation[0]["estimated_cost_micros"]
    planner = r.seed.app.state.inference.chain_planner
    loader = planner.loader
    calls = 0

    class MixedLoader:
        def load(self, version, identity):
            candidate = loader.load(version, identity)

            class MixedProvider:
                async def generate(self, request):
                    nonlocal calls
                    calls += 1
                    result = await candidate.provider.generate(request)
                    return replace(result, content="") if calls % 5 == 0 else result

            return replace(candidate, provider=MixedProvider())

    planner.loader = MixedLoader()
    results = [infer(r) for _ in range(10)]
    assert sum(response["route"]["fallback_used"] for response in results) == 2
    assert sum(response["route"]["specialist_served"] for response in results) == 8
    total = sum(response["estimated_cost_micros"] for response in results)
    costs = []
    for response in results:
        interaction = r.seed.app.state.metadata.get(
            "synthetic-a", Interaction, response["interaction_id"]
        )
        attempts = [
            r.seed.app.state.metadata.get("synthetic-a", GenerationAttempt, attempt_id)
            for attempt_id in interaction.generation_attempt_ids
        ]
        assert sum(a.estimated_cost_micros for a in attempts) == interaction.total_cost_micros
        costs.append(attempts[0].estimated_cost_micros)
    assert len(set(costs)) == 1
    assert total == 10 * costs[0] + 2 * foundation_unit_cost
    report = r.seed.client.get(
        "/v1/canary/reports", headers=OPERATOR, params={"policy": p.policy_id, "since": p_time()}
    ).json()
    aggregate = report["overall"]
    assert aggregate["specialist"]["total_cost_micros"] == total
    assert aggregate["foundation"]["total_cost_micros"] == 10 * foundation_unit_cost
    assert aggregate["cost_reduction_fraction"] == pytest.approx(
        1 - total / (10 * foundation_unit_cost)
    )
    assert aggregate["cost_delta_micros"]["mean_delta"] == pytest.approx(
        total / 10 - foundation_unit_cost
    )
    assert aggregate["foundation_served"]["interactions"] == 12
    assert aggregate["shadow_cost_micros"] > 0
    assert report["passed"] and aggregate["cost_reduction_fraction"] >= 0.3
    print(
        f"cost-exit-live: attempts=12; specialist_total_micros={total}; successes=10; "
        f"foundation_per_success={foundation_unit_cost}; "
        f"reduction={aggregate['cost_reduction_fraction']:.3f}; fallback_rate=0.2"
    )


@pytest.mark.parametrize(
    "metric,value", [("false_specialist_rate", 0.021), ("calibration_error", 0.101)]
)
def test_router_promotion_rechecks_failed_gate(router_seed, monkeypatch, metric, value):
    from conftest import wait_training

    from adaptive_llm.routing.model import routing_suite

    r = router_seed
    spec = TrainingJobSpecification(
        job_type="router",
        registry_id="synthetic-router",
        dataset_id=r.dataset.dataset_id,
        dataset_version=r.dataset.version,
    )
    assert (
        r.seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
        ).status_code
        == 200
    )
    job = wait_training(r.seed.client, spec.job_id, OPERATOR)
    assert job.state == "succeeded"

    def failing_suite(*args):
        result = routing_suite(*args)
        return result.model_copy(update={"metrics": {**result.metrics, metric: value}})

    monkeypatch.setattr("adaptive_llm.routing.model.routing_suite", failing_suite)
    request = EvaluationInput(
        candidate_deployment_id=job.model_version,
        baseline_deployment_id=None,
        dataset_id=r.dataset.dataset_id,
        dataset_version=r.dataset.version,
        suites=["routing"],
        minimum_sample_size=4,
    )
    report = r.seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=request.model_dump(mode="json")
    )
    assert report.status_code == 200 and not report.json()["passed"]
    result = r.seed.client.post(
        f"/v1/models/{job.model_version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": job.model_version,
            "target_state": "approved",
            "evaluation_id": request.evaluation_id,
            **NOTE,
        },
    )
    assert result.status_code == 409
    assert result.json()["error"]["code"] == "passed_evaluation_required"
