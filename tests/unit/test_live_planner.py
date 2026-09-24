from dataclasses import replace
from unittest.mock import Mock

import pytest

from adaptive_llm.contracts import (
    AdapterConfig,
    CanaryConfig,
    ModelManifest,
    PolicyDecision,
    RoutePolicy,
    RoutingFeatures,
    RoutingObservation,
    RoutingOptions,
    RoutingRow,
    uid,
)
from adaptive_llm.metrics import InProcessMetrics
from adaptive_llm.providers import FakeProvider
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.routing import FoundationRouter
from adaptive_llm.routing.breakers import CircuitBreakers
from adaptive_llm.routing.chain import ExecutionCandidate
from adaptive_llm.routing.control import RouteSnapshot
from adaptive_llm.routing.live import LivePlanner
from adaptive_llm.routing.model import train_model


@pytest.fixture
def planner_setup(settings, provider_request, identity, keyring, tmp_path, monkeypatch):
    foundation = FoundationRouter(settings.routing_path)
    features = RoutingFeatures(
        task="general",
        input_tokens=provider_request.input_tokens,
        context_supplied=bool(provider_request.context),
    )
    versions = [uid() for _ in range(4)]
    policy = RoutePolicy(
        eligible_specialist_versions=versions[:3],
        live_specialists_allowed=True,
        router_version=versions[3],
        canary=CanaryConfig(traffic_fraction=1),
        tenant_enabled={identity.tenant_id: True},
        task_enabled={"general": True},
    )
    models = {
        version: ModelManifest(
            registry_id="synthetic",
            version=version,
            state="production",
            tenant_ids=[identity.tenant_id],
            base_model_id="synthetic",
            base_model_revision="synthetic",
            base_model_licence="synthetic",
            adapter_config=AdapterConfig(),
            tokenizer_id="synthetic",
            chat_template_version="synthetic",
            datasets=[],
            training_job_id=uid(),
            code_revision="synthetic",
            container_digest="local",
            configuration_digest="synthetic",
            seed=23,
            hardware_class="cpu",
            artifact_hashes={},
            artifact_digest="synthetic",
            artifact_mac="synthetic",
            storage_location="synthetic",
            input_micros_per_1000_tokens=(100, 200, 50, 0)[index],
            output_micros_per_1000_tokens=(100, 200, 50, 0)[index],
            adapter_architecture="router-logistic-v1"
            if index == 3
            else "deterministic-fake-adapter-v1",
        )
        for index, version in enumerate(versions)
    }
    observation = RoutingObservation(quality=1, validation_pass=True, cost_micros=10, latency_ms=1)
    rows = [
        RoutingRow(
            interaction_id=f"synthetic-{i}",
            tenant_id=identity.tenant_id,
            features=features,
            source_dataset_id="synthetic",
            source_dataset_version="synthetic",
            foundation_id="foundation",
            candidates={
                "foundation": observation,
                versions[0]: observation,
                versions[1]: observation,
                versions[2]: None,
            },
        )
        for i in range(40)
    ]
    router = train_model(rows[:30], rows[30:], lambda: None)
    monkeypatch.setattr(
        "adaptive_llm.routing.live.verify_artifact",
        Mock(return_value={"router.json": router.model_dump_json().encode()}),
    )
    registry = Mock(spec=ModelRegistry)
    registry.get.side_effect = lambda version, _: models[version]
    store = Mock()
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    loader = Mock()
    loader.load.return_value = ExecutionCandidate(
        foundation.deployment, FakeProvider(), specialist=True
    )
    metrics = InProcessMetrics()
    breakers = CircuitBreakers(metrics)
    planner = LivePlanner(
        store,
        FakeProvider(),
        metrics,
        registry,
        loader,
        keyring,
        tmp_path,
        frozenset({identity.tenant_id}),
        breakers,
    )
    decision = PolicyDecision(policy_version="synthetic", retention_seconds=1)
    selection = replace(
        foundation.route(provider_request, RoutingOptions(), decision, uid()),
        request=provider_request,
        features=features,
        policy=decision,
    )
    return planner, selection, identity, models, versions, store, loader, policy


def test_cheapest_qualifying_model_and_all_candidate_estimates(planner_setup):
    planner, selection, identity, _, versions, _, _, _ = planner_setup
    plan = planner.plan(selection, RoutingOptions(mode="specialist"), identity, "general")
    assert len(plan.candidates) == 2
    assert plan.candidates[0].deployment.model_deployment_id == versions[0]
    assert (
        plan.candidates[-1].deployment.model_deployment_id
        == selection.deployment.model_deployment_id
    )
    assert len(plan.considered) == 4
    by_id = {c.deployment.model_deployment_id: c for c in plan.considered}
    assert by_id[versions[0]].quality > 0.99
    assert by_id[versions[1]].rejection_reason == "not_selected"
    assert by_id[versions[2]].rejection_reason == "low_quality"
    assert by_id[versions[2]].quality < 0.01
    assert (
        len(
            planner.plan(
                selection, RoutingOptions(mode="foundation"), identity, "general"
            ).candidates
        )
        == 1
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"state": "shadow"}, "state_ineligible"),
        ({"processing_region": "eu-west"}, "policy_uncertainty"),
        ({"tenant_ids": ["synthetic-other"]}, "policy_uncertainty"),
        ({"context_limit": 1}, "context_length"),
        ({"modalities": []}, "unsupported_modality"),
    ],
)
def test_hard_constraints_before_provider_loading(planner_setup, change, reason):
    planner, selection, identity, models, versions, _, loader, _ = planner_setup
    for version in versions[:3]:
        models[version] = models[version].model_copy(update=change)
    plan = planner.plan(selection, RoutingOptions(), identity, "general")
    assert len(plan.candidates) == 1
    assert all(c.rejection_reason == reason for c in plan.considered[:-1])
    assert all(not c.has_estimate for c in plan.considered[:-1])
    loader.load.assert_not_called()


def test_disablement_allowlist_and_control_failure_fail_closed(planner_setup):
    planner, selection, identity, _, versions, store, loader, policy = planner_setup
    store.snapshot.return_value = RouteSnapshot(
        policy=policy, disabled_specialists=frozenset(versions)
    )
    assert len(planner.plan(selection, RoutingOptions(), identity, "general").candidates) == 1
    loader.load.assert_not_called()
    policy = policy.model_copy(
        update={
            "canary": CanaryConfig(
                traffic_fraction=1,
                tenant_allowlist=["synthetic-denied"],
            )
        }
    )
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    assert (
        planner.plan(selection, RoutingOptions(), identity, "general").fallback_reason
        == "canary_not_assigned"
    )
    store.snapshot.side_effect = RuntimeError("synthetic")
    assert len(planner.plan(selection, RoutingOptions(), identity, "general").candidates) == 1


def test_router_and_admission_manifests_share_snapshot_cache(planner_setup):
    from adaptive_llm.routing.live import verify_artifact

    planner, selection, identity, _, versions, store, _, policy = planner_setup
    for _ in range(10):
        assert len(planner.plan(selection, RoutingOptions(), identity, "general").candidates) == 2
    assert planner.registry.get.call_count == len(versions) + 1  # cold-load recheck
    assert verify_artifact.call_count == 1
    # A fresh pointer snapshot also refreshes the manifests even if policy values are equal.
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    planner.plan(selection, RoutingOptions(), identity, "general")
    assert planner.registry.get.call_count == 2 * len(versions) + 1
    assert verify_artifact.call_count == 1
    # Cached admission never skips the current caller's tenant membership.
    denied = replace(identity, tenant_id="synthetic-other")
    policy = policy.model_copy(update={"tenant_enabled": {denied.tenant_id: True}})
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    assert len(planner.plan(selection, RoutingOptions(), denied, "general").candidates) == 1
    assert verify_artifact.call_count == 1


@pytest.mark.parametrize("change", [{"state": "revoked"}, {"artifact_digest": "synthetic-new"}])
def test_cache_invalidates_router_on_registry_change(planner_setup, change):
    from adaptive_llm.routing.live import verify_artifact

    planner, selection, identity, models, versions, store, _, policy = planner_setup
    planner.plan(selection, RoutingOptions(), identity, "general")
    old_key = (versions[-1], models[versions[-1]].artifact_digest)
    models[versions[-1]] = models[versions[-1]].model_copy(update=change)
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    plan = planner.plan(selection, RoutingOptions(), identity, "general")
    assert old_key not in planner._routers
    if "state" in change:
        assert len(plan.candidates) == 1
        assert verify_artifact.call_count == 1
    else:
        assert len(plan.candidates) == 2
        assert verify_artifact.call_count == 2


def test_specialist_admission_refreshes_state_and_digest(planner_setup):
    planner, selection, identity, models, versions, store, _, policy = planner_setup
    assert (
        planner.plan(selection, RoutingOptions(), identity, "general")
        .candidates[0]
        .deployment.model_deployment_id
        == versions[0]
    )
    models[versions[0]] = models[versions[0]].model_copy(update={"state": "revoked"})
    models[versions[1]] = models[versions[1]].model_copy(
        update={"artifact_digest": "synthetic-new", "context_limit": 1}
    )
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    plan = planner.plan(selection, RoutingOptions(), identity, "general")
    assert len(plan.candidates) == 1
    reasons = {c.deployment.model_deployment_id: c.rejection_reason for c in plan.considered}
    assert reasons[versions[0]] == "state_ineligible"
    assert reasons[versions[1]] == "context_length"
    assert planner._manifests[versions[1]].artifact_digest == "synthetic-new"


def test_registry_failure_evicts_cache_and_counts_degradation(planner_setup):
    planner, selection, identity, _, _, store, _, policy = planner_setup
    planner.plan(selection, RoutingOptions(), identity, "general")
    store.snapshot.return_value = RouteSnapshot(policy=policy)
    planner.registry.get.side_effect = RuntimeError("synthetic")
    plan = planner.plan(selection, RoutingOptions(), identity, "general")
    assert len(plan.candidates) == 1 and plan.fallback_reason == "policy_uncertainty"
    assert planner.metrics.get("live_planner_failures") == 1
    assert not planner._routers and not planner._manifests


@pytest.mark.parametrize("change", [{"state": "revoked"}, {"artifact_digest": "synthetic-new"}])
def test_registry_change_during_verification_never_enters_cache(planner_setup, monkeypatch, change):
    from adaptive_llm.routing.live import verify_artifact

    planner, selection, identity, models, versions, _, _, _ = planner_setup

    def verify(*args):
        models[versions[-1]] = models[versions[-1]].model_copy(update=change)
        return verify_artifact(*args)

    monkeypatch.setattr("adaptive_llm.routing.live.verify_artifact", verify)
    plan = planner.plan(selection, RoutingOptions(), identity, "general")
    assert len(plan.candidates) == 1 and plan.fallback_reason == "policy_uncertainty"
    assert planner.metrics.get("live_planner_failures") == 1
    assert not planner._routers


def test_more_expensive_specialist_uses_not_cheapest_reason(planner_setup):
    planner, selection, identity, models, versions, store, loader, policy = planner_setup
    store.snapshot.return_value = RouteSnapshot(
        policy=policy.model_copy(update={"eligible_specialist_versions": versions[:2]})
    )
    for version in versions[:2]:
        models[version] = models[version].model_copy(
            update={"input_micros_per_1000_tokens": 10_000, "output_micros_per_1000_tokens": 10_000}
        )
    plan = planner.plan(selection, RoutingOptions(), identity, "general")
    assert len(plan.candidates) == 1
    assert plan.fallback_reason == "not_cheapest"
    by_id = {c.deployment.model_deployment_id: c for c in plan.considered}
    assert all(by_id[version].rejection_reason == "not_cheapest" for version in versions[:2])
    loader.load.assert_not_called()
