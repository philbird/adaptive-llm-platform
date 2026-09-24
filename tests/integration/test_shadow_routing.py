import asyncio
import json
from dataclasses import replace
from threading import Event as ThreadEvent
from time import monotonic, sleep
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    Citation,
    GenerationAttempt,
    InferenceRequest,
    Interaction,
    PolicyDecision,
    RouteDecision,
    RoutePolicy,
    RoutingOptions,
    now,
    uid,
)
from adaptive_llm.gateway.identity import Identity
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.routing import FoundationRouter, RouteSelection
from adaptive_llm.routing.chain import ChainPlan, ExecutionCandidate

if TYPE_CHECKING:
    from conftest import EvaluationSeed

OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}
USER = {"Authorization": "Bearer synthetic-key-a"}
NOTE = {"reason": "SYNTHETIC route control"}


def activate(client: TestClient, version: str, **updates: object) -> RoutePolicy:
    policy = RoutePolicy.model_validate(
        {
            "eligible_specialist_versions": [version],
            "shadow_enabled": True,
            "tenant_enabled": {"synthetic-a": True},
            "task_enabled": {"question_answering": True, "general": True},
            **updates,
        }
    )
    response = client.post(
        "/v1/route-policies", headers=OPERATOR, json=policy.model_dump(mode="json")
    )
    assert response.status_code == 200, response.json()
    assert (
        client.post(
            f"/v1/route-policies/{policy.policy_id}/activate",
            headers=OPERATOR,
            json=NOTE,
        ).status_code
        == 200
    )
    return policy


def infer(client: TestClient, request: InferenceRequest) -> dict[str, object]:
    response = client.post("/v1/inference", headers=USER, json=request.model_dump(mode="json"))
    assert response.status_code == 200, response.json()
    return response.json()


def drain(seed: "EvaluationSeed") -> None:
    assert seed.client.portal is not None
    seed.client.portal.call(seed.app.state.shadow.queue.join)


def shadow_attempts(seed: "EvaluationSeed", iid: str) -> list[GenerationAttempt]:
    with seed.app.state.database.lock:
        rows = seed.app.state.database.connection.execute(
            "SELECT data FROM attempts WHERE interaction_id=?",
            (iid,),
        ).fetchall()
    return [a for r in rows if (a := GenerationAttempt.model_validate_json(r[0])).shadow]


def test_real_registry_shadow_preserves_response_replay_cost_and_canonical_input(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
) -> None:
    seed, version = shadow_seed
    before = infer(seed.client, inference_request)
    policy = activate(seed.client, version)
    loader = seed.app.state.shadow.loader
    requests: list[ProviderRequest] = []

    class CapturingLoader:
        def load(self, version: str, identity: Identity) -> ExecutionCandidate:
            candidate = loader.load(version, identity)

            class CapturingProvider:
                async def generate(self, request: ProviderRequest) -> ProviderResult:
                    requests.append(request)
                    return await candidate.provider.generate(request)

            return replace(candidate, provider=CapturingProvider())

    seed.app.state.shadow.loader = CapturingLoader()
    request = inference_request.model_copy(update={"request_id": uid()})
    first = infer(seed.client, request)
    drain(seed)
    assert requests and requests[0].messages == tuple(request.messages)
    assert all(c.tenant_id == "synthetic-a" and c.region == "local" for c in requests[0].context)
    for field in ["content", "usage", "estimated_cost_micros", "citations", "model_deployment_id"]:
        assert first[field] == before[field]
    attempts = shadow_attempts(seed, first["interaction_id"])
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.attempt_number == 0 and attempt.output_ref
    assert attempt.validation.passed
    assert seed.app.state.metrics.get("shadow_cost_micros") == attempt.estimated_cost_micros
    interaction = seed.app.state.metadata.get("synthetic-a", Interaction, first["interaction_id"])
    assert attempt.attempt_id not in interaction.generation_attempt_ids
    assert attempt.attempt_id != interaction.final_attempt_id
    replay = infer(seed.client, request)
    assert replay == {**first, "replayed": True}
    drain(seed)
    assert len(shadow_attempts(seed, first["interaction_id"])) == 1
    rows = seed.app.state.database.connection.execute(
        "SELECT envelope FROM outbox WHERE event_type='generation.completed.v1'"
    ).fetchall()
    assert any(json.loads(r[0])["data"].get("shadow") is True for r in rows)
    report = seed.client.get(
        "/v1/shadow/reports",
        headers=OPERATOR,
        params={"policy": policy.policy_id, "since": "2000-01-01T00:00:00Z"},
    )
    assert report.status_code == 200
    summary = report.json()["overall"]
    assert summary["coverage"] == summary["specialist_validation_pass_rate"] == 1
    assert summary["score_delta"]["sample_size"] == 1
    assert summary["score_delta"]["mean_delta"] == 0
    assert summary["mean_cost_delta_micros"] == 0
    assert "citation" in report.json()["critical_segments"]
    assert str(first["content"]) not in report.text
    assert requests[0].messages[0].content not in report.text
    assert "interaction_id" not in report.text
    assert (
        seed.client.get(
            "/v1/shadow/reports",
            headers=USER,
            params={"policy": policy.policy_id, "since": now().isoformat()},
        ).status_code
        == 403
    )


def test_shadow_reuses_provider_then_revocation_evicts_it(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, version = shadow_seed
    activate(seed.client, version)
    loaded: list[SpecialistProvider] = []

    def load_provider(*args: object, **kwargs: object) -> SpecialistProvider:
        provider = SpecialistProvider(*args, **kwargs)
        loaded.append(provider)
        return provider

    monkeypatch.setattr("adaptive_llm.routing.shadow.SpecialistProvider", load_provider)
    first = infer(seed.client, inference_request)
    drain(seed)
    second = infer(seed.client, inference_request.model_copy(update={"request_id": uid()}))
    drain(seed)
    assert len(loaded) == 1
    assert shadow_attempts(seed, first["interaction_id"])
    assert shadow_attempts(seed, second["interaction_id"])
    loader = seed.app.state.shadow.loader
    assert len(loader._cache) == 1
    assert next(iter(loader._cache.values())).provider is loaded[0]
    assert (
        seed.client.post(
            f"/v1/models/{version}/promotion-requests",
            headers=OPERATOR,
            json={"model_version": version, "target_state": "revoked", **NOTE},
        ).status_code
        == 200
    )
    third = infer(seed.client, inference_request.model_copy(update={"request_id": uid()}))
    drain(seed)
    assert not loader._cache
    assert len(loaded) == 1
    assert not shadow_attempts(seed, third["interaction_id"])


@pytest.mark.parametrize("logging", [True, False])
def test_shadow_private_output_never_served_or_replayed(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
    logging: bool,
) -> None:
    seed, version = shadow_seed
    seed.app.state.training.policy.logging = logging
    activate(seed.client, version)
    foundation = FoundationRouter(Settings().routing_path).deployment

    class PrivateProvider:
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            return replace(
                await FakeProvider().generate(request), content="SYNTHETIC_SHADOW_PRIVATE"
            )

    class Loader:
        def load(self, version: str, identity: Identity) -> ExecutionCandidate:
            return ExecutionCandidate(
                foundation.model_copy(update={"model_deployment_id": version}),
                PrivateProvider(),
                specialist=True,
            )

    seed.app.state.shadow.loader = Loader()
    result = infer(seed.client, inference_request)
    drain(seed)
    attempt = shadow_attempts(seed, result["interaction_id"])[0]
    assert attempt.error_code is None
    assert attempt.validation.passed
    grounding = next(c for c in attempt.validation.checks if c.name == "groundedness")
    assert not grounding.passed and grounding.severity == "advisory"
    assert (
        seed.app.state.metrics.get(
            "validation_advisory_failures",
            check_name="groundedness",
        )
        == 1
    )
    comparison = json.loads(
        seed.app.state.evaluation_database.connection.execute(
            "SELECT data FROM shadow_comparisons WHERE interaction_id=?",
            (result["interaction_id"],),
        ).fetchone()[0]
    )
    assert comparison["judge_version"] == "shadow-chunk-overlap-1"
    assert {c["severity"] for c in comparison["specialist_validation"]["checks"]} == {
        "hard",
        "advisory",
    }
    assert bool(attempt.output_ref) is logging
    replay = infer(seed.client, inference_request)
    assert "SYNTHETIC_SHADOW_PRIVATE" not in json.dumps([result, replay])
    assert "SYNTHETIC_SHADOW_PRIVATE" not in str(
        list(seed.app.state.database.connection.execute("SELECT data FROM attempts"))
    )
    assert attempt.output_hash
    if logging:
        blob = seed.app.state.payloads.get("synthetic-a", attempt.output_ref, now())
        assert (
            seed.app.state.persistence.cipher.decrypt(
                blob,
                "synthetic-a",
                result["interaction_id"],
                f"shadow_output.{attempt.attempt_id}",
            )
            == b"SYNTHETIC_SHADOW_PRIVATE"
        )


def test_open_breaker_skips_and_queue_is_bounded(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
) -> None:
    seed, version = shadow_seed
    policy = activate(seed.client, version)
    breakers = seed.app.state.inference.breakers
    for _ in range(5):
        breakers.record(version, policy.breaker, error=True, validation_failure=False, latency_ms=1)
    first = infer(seed.client, inference_request)
    drain(seed)
    assert not shadow_attempts(seed, first["interaction_id"])
    assert seed.app.state.metrics.get("shadow_drops", reason="circuit_open") == 1
    # A separate eligible deployment avoids resetting breaker history.
    assert seed.client.get("/healthz").json()["circuit_breakers"][version] == "open"


def test_post_response_worker_never_blocks_and_full_queue_drops(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
) -> None:
    seed, version = shadow_seed
    activate(seed.client, version)
    loader = seed.app.state.shadow.loader
    entered, release = ThreadEvent(), ThreadEvent()

    class SlowLoader:
        def load(self, version: str, identity: Identity) -> ExecutionCandidate:
            candidate = loader.load(version, identity)

            class SlowProvider:
                async def generate(self, request: ProviderRequest) -> ProviderResult:
                    entered.set()
                    assert release.wait(10)
                    return await candidate.provider.generate(request)

            return replace(candidate, provider=SlowProvider())

    seed.app.state.shadow.loader = SlowLoader()
    seed.app.state.shadow.queue = asyncio.Queue(maxsize=1)
    try:
        first = infer(seed.client, inference_request)
        assert entered.wait(2)
        assert not shadow_attempts(seed, first["interaction_id"])
        infer(seed.client, inference_request.model_copy(update={"request_id": uid()}))
        infer(seed.client, inference_request.model_copy(update={"request_id": uid()}))
        assert seed.app.state.shadow.queue.qsize() == 1
        assert seed.app.state.metrics.get("shadow_drops", reason="queue_full") == 1
    finally:
        release.set()
    drain(seed)
    assert seed.app.state.metrics.get("shadow_completed") == 2


def test_policy_immutable_activation_rollback_and_public_live_rejection(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
) -> None:
    seed, version = shadow_seed
    first = activate(seed.client, version)
    second = activate(seed.client, version, shadow_enabled=False)
    assert first.policy_id != second.policy_id
    assert (
        seed.client.post(
            "/v1/route-policies", headers=OPERATOR, json=first.model_dump(mode="json")
        ).status_code
        == 409
    )
    assert (
        seed.client.post(
            "/v1/route-policies",
            headers=OPERATOR,
            json={
                **first.model_dump(mode="json"),
                "policy_id": uid(),
                "live_specialists_allowed": True,
            },
        ).status_code
        == 422
    )
    assert (
        seed.client.post(
            f"/v1/route-policies/{first.policy_id}/activate",
            headers=USER,
            json=NOTE,
        ).status_code
        == 403
    )
    assert (
        seed.client.post(
            f"/v1/route-policies/{first.policy_id}/activate",
            headers=OPERATOR,
            json={"reason": " "},
        ).status_code
        == 422
    )
    with seed.app.state.evaluation_database.lock:
        before = seed.app.state.evaluation_database.connection.execute(
            "SELECT data FROM model_versions WHERE version=?",
            (version,),
        ).fetchone()[0]
    assert (
        seed.client.post(
            f"/v1/route-policies/{first.policy_id}/activate",
            headers=OPERATOR,
            json=NOTE,
        ).status_code
        == 200
    )
    with seed.app.state.evaluation_database.lock:
        assert (
            seed.app.state.evaluation_database.connection.execute(
                "SELECT data FROM model_versions WHERE version=?",
                (version,),
            ).fetchone()[0]
            == before
        )
    result = infer(
        seed.client,
        inference_request.model_copy(
            update={
                "routing": inference_request.routing.model_copy(update={"mode": "specialist"}),
            }
        ),
    )
    assert result["model_deployment_id"] == "fake-foundation-local-1"
    interaction = seed.app.state.metadata.get("synthetic-a", Interaction, result["interaction_id"])
    route = seed.app.state.metadata.get("synthetic-a", RouteDecision, interaction.route_decision_id)
    assert route.fallback_reasons == ["live_specialists_disabled"]
    assert seed.app.state.metrics.get("fallback_reasons", reason="live_specialists_disabled") == 1
    drain(seed)
    # Registry state is rechecked when activating an existing policy.
    assert (
        seed.client.post(
            f"/v1/models/{version}/promotion-requests",
            headers=OPERATOR,
            json={"model_version": version, "target_state": "canary", **NOTE},
        ).status_code
        == 200
    )
    assert (
        seed.client.post(
            f"/v1/route-policies/{first.policy_id}/activate",
            headers=OPERATOR,
            json=NOTE,
        ).status_code
        == 409
    )
    result = infer(seed.client, inference_request.model_copy(update={"request_id": uid()}))
    drain(seed)
    assert not shadow_attempts(seed, result["interaction_id"])


@pytest.mark.parametrize(
    "scope", [{"tenant_id": "synthetic-a"}, {"task": "question_answering"}, {}]
)
def test_persistent_disablement_and_reenable(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
    scope: dict[str, str],
) -> None:
    seed, version = shadow_seed
    activate(seed.client, version)
    assert (
        seed.client.post(
            "/v1/route-policies/kill-switch", headers=OPERATOR, json={**NOTE, **scope}
        ).status_code
        == 200
    )
    first = infer(seed.client, inference_request)
    drain(seed)
    assert not shadow_attempts(seed, first["interaction_id"])
    assert (
        seed.client.request(
            "DELETE", "/v1/route-policies/kill-switch", headers=OPERATOR, json={**NOTE, **scope}
        ).status_code
        == 200
    )
    second = infer(seed.client, inference_request.model_copy(update={"request_id": uid()}))
    drain(seed)
    assert shadow_attempts(seed, second["interaction_id"])


def test_kill_switch_across_two_instances_without_restart(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
) -> None:
    seed, version = shadow_seed
    activate(seed.client, version)
    app = create_app(Settings(data_dir=seed.directory, outbox_dispatch_enabled=False))
    with TestClient(app) as second:
        assert not second.get("/healthz").json()["kill_switch"]
        started = monotonic()
        assert (
            seed.client.post(
                "/v1/route-policies/kill-switch", headers=OPERATOR, json=NOTE
            ).status_code
            == 200
        )
        while not second.get("/healthz").json()["kill_switch"]:
            assert monotonic() - started < 2
            sleep(0.01)
        assert monotonic() - started < 2
        result = infer(second, inference_request)
        assert second.portal is not None
        second.portal.call(app.state.shadow.queue.join)
        assert not shadow_attempts(seed, result["interaction_id"])
        assert app.state.metrics.get("shadow_drops", reason="disabled") == 1
    restarted = create_app(Settings(data_dir=seed.directory, outbox_dispatch_enabled=False))
    with TestClient(restarted) as client:
        assert client.get("/healthz").json()["kill_switch"]


def test_injected_live_chain_records_both_attempts_and_replays_only_foundation(
    inference_request: InferenceRequest,
) -> None:
    class InvalidSpecialist:
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            return replace(
                await FakeProvider().generate(request),
                content="SYNTHETIC_FAILED_SPECIALIST",
                citations=(Citation(document_id="synthetic-invented", chunk_id="missing"),),
            )

    class InjectedPlanner:
        def plan(
            self,
            selection: RouteSelection,
            options: RoutingOptions,
            identity: Identity,
            task: str,
        ) -> ChainPlan:
            return ChainPlan(
                (
                    ExecutionCandidate(
                        selection.deployment.model_copy(
                            update={"model_deployment_id": "synthetic-live"}
                        ),
                        InvalidSpecialist(),
                        specialist=True,
                    ),
                    ExecutionCandidate(selection.deployment, FakeProvider()),
                ),
                RoutePolicy(),
                allow_live_specialists=True,
            )

    app = create_app(Settings(chain_planner=InjectedPlanner(), outbox_dispatch_enabled=False))
    with TestClient(app) as client:
        result = infer(client, inference_request)
        assert result["model_deployment_id"] == "fake-foundation-local-1"
        assert result["route"]["fallback_used"]
        attempts = [
            GenerationAttempt.model_validate_json(r[0])
            for r in app.state.database.connection.execute(
                "SELECT data FROM attempts ORDER BY rowid"
            )
        ]
        assert [a.attempt_number for a in attempts] == [1, 2]
        assert attempts[0].error_code == "validation_failed"
        assert attempts[0].output_ref is None
        assert attempts[1].fallback_reason == "validation_failure"
        assert result["estimated_cost_micros"] == sum(a.estimated_cost_micros for a in attempts)
        interaction = app.state.metadata.get("synthetic-a", Interaction, result["interaction_id"])
        assert interaction.final_attempt_id == attempts[1].attempt_id
        assert interaction.generation_attempt_ids == [a.attempt_id for a in attempts]
        route = app.state.metadata.get("synthetic-a", RouteDecision, interaction.route_decision_id)
        assert [c.model_deployment_id for c in route.candidates] == [
            "synthetic-live",
            "fake-foundation-local-1",
        ]
        assert "SYNTHETIC_FAILED_SPECIALIST" not in json.dumps(infer(client, inference_request))


@pytest.mark.parametrize("failure", ["redaction", "deletion", "policy_revocation", "provider"])
def test_shadow_failures_preserve_serving_and_fail_closed_for_content(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
    failure: str,
) -> None:
    seed, version = shadow_seed
    activate(seed.client, version)
    loader = seed.app.state.shadow.loader
    entered, release = ThreadEvent(), ThreadEvent()
    redactor = seed.app.state.persistence.redactor

    class ShadowRedactor:
        version = "synthetic-failing-shadow-1"

        def redact_text(self, text: str, policy: PolicyDecision) -> tuple[str, dict[str, int]]:
            if "SYNTHETIC_SHADOW_PRIVATE" in text:
                raise ValueError("synthetic_redaction_failure")
            return redactor.redact_text(text, policy)

    class Loader:
        def load(self, version: str, identity: Identity) -> ExecutionCandidate:
            candidate = loader.load(version, identity)

            class ShadowProvider:
                async def generate(self, request: ProviderRequest) -> ProviderResult:
                    entered.set()
                    assert release.wait(10)
                    if failure == "provider":
                        raise RuntimeError("SYNTHETIC_PRIVATE_PROVIDER_ERROR")
                    return replace(
                        await candidate.provider.generate(request),
                        content="SYNTHETIC_SHADOW_PRIVATE",
                    )

            return replace(candidate, provider=ShadowProvider())

    seed.app.state.shadow.loader = Loader()
    if failure == "redaction":
        seed.app.state.persistence.redactor = ShadowRedactor()
    try:
        result = infer(seed.client, inference_request)
        assert entered.wait(2)
        if failure == "deletion":
            assert (
                seed.client.delete(
                    f"/v1/privacy/interactions/{result['interaction_id']}",
                    headers=USER,
                ).status_code
                == 204
            )
        elif failure == "policy_revocation":
            seed.app.state.training.policy.logging = False
    finally:
        release.set()
    drain(seed)
    attempts = shadow_attempts(seed, result["interaction_id"])
    assert "SYNTHETIC_SHADOW_PRIVATE" not in str(result)
    if failure == "deletion":
        assert not attempts
    else:
        assert len(attempts) == 1
        assert attempts[0].output_ref is None
        if failure == "redaction":
            assert attempts[0].output_hash is None
        if failure == "provider":
            assert attempts[0].error_code == "provider_failed"
            assert "SYNTHETIC_PRIVATE_PROVIDER_ERROR" not in str(attempts)
        assert infer(seed.client, inference_request) == {**result, "replayed": True}


def test_foundation_breaker_excludes_candidate_and_control_outage_preserves_serving(
    inference_request: InferenceRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(Settings(outbox_dispatch_enabled=False))
    with TestClient(app) as client:

        def unavailable() -> None:
            raise RuntimeError("synthetic_control_unavailable")

        monkeypatch.setattr(app.state.route_policies, "snapshot", unavailable)
        first = infer(client, inference_request)
        assert first["model_deployment_id"] == "fake-foundation-local-1"
        assert client.get("/healthz").json()["kill_switch"]
        breakers = app.state.inference.breakers
        for _ in range(5):
            breakers.record(
                "fake-foundation-local-1",
                RoutePolicy().breaker,
                error=True,
                validation_failure=False,
                latency_ms=1,
            )
        second = client.post(
            "/v1/inference",
            headers=USER,
            json=inference_request.model_copy(update={"request_id": uid()}).model_dump(mode="json"),
        )
        assert second.status_code == 503
        last = app.state.database.connection.execute(
            "SELECT data FROM route_decisions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        route = RouteDecision.model_validate_json(last)
        assert route.selected_model_deployment_id is None
        assert not route.candidates[0].eligible
        assert route.candidates[0].reason_codes == ["circuit_open"]
        assert app.state.metrics.get("fallback_reasons", reason="circuit_open") == 1
