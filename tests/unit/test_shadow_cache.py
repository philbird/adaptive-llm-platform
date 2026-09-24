from pathlib import Path
from unittest.mock import Mock

import pytest

from adaptive_llm.app import Settings
from adaptive_llm.contracts import AdapterConfig, ModelManifest, uid
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.providers import FakeProvider
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.routing import FoundationRouter
from adaptive_llm.routing.shadow import RegistrySpecialists


def manifest() -> ModelManifest:
    return ModelManifest(
        registry_id="synthetic-cache",
        version=uid(),
        state="shadow",
        tenant_ids=["synthetic-a"],
        base_model_id="synthetic-base",
        base_model_revision="synthetic-1",
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
        artifact_digest="synthetic-digest",
        artifact_mac="synthetic",
        storage_location="synthetic",
    )


def test_cache_is_lru_bounded_and_rechecks_registry(
    settings: Settings,
    keyring: Keyring,
    identity: Identity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [manifest() for _ in range(9)]
    current = {m.version: m for m in models}
    registry = Mock(spec=ModelRegistry)
    registry.get.side_effect = lambda version, _: current[version]
    loaded: list[ModelManifest] = []

    class Provider(FakeProvider):
        def __init__(self, model: ModelManifest, *args: object) -> None:
            super().__init__()
            self.model_version = model.artifact_digest
            loaded.append(model)

    monkeypatch.setattr("adaptive_llm.routing.shadow.SpecialistProvider", Provider)
    loader = RegistrySpecialists(
        registry,
        FoundationRouter(settings.routing_path).deployment,
        keyring,
        tmp_path,
        frozenset({"synthetic-a"}),
    )
    candidates = [loader.load(m.version, identity) for m in models[:8]]
    assert loader.load(models[0].version, identity) is candidates[0]
    assert len(loaded) == 8
    loader.load(models[8].version, identity)
    assert len(loader._cache) == 8
    assert loader.load(models[1].version, identity).provider is not candidates[1].provider
    assert len(loaded) == 10
    first = models[0]
    current[first.version] = first.model_copy(update={"artifact_digest": "synthetic-new-digest"})
    replacement = loader.load(first.version, identity)
    assert replacement.provider is not candidates[0].provider
    assert (first.version, first.artifact_digest) not in loader._cache
    assert (first.version, "synthetic-new-digest") in loader._cache
    # Cached weights never bypass tenant membership checks.
    current[first.version] = current[first.version].model_copy(
        update={"tenant_ids": ["synthetic-b"]}
    )
    with pytest.raises(GatewayError, match="shadow_model_ineligible"):
        loader.load(first.version, identity)
    current[first.version] = current[first.version].model_copy(update={"state": "revoked"})
    with pytest.raises(GatewayError, match="shadow_model_ineligible"):
        loader.load(first.version, identity)
    assert not any(key[0] == first.version for key in loader._cache)
    registry.get.side_effect = GatewayError(404, "model_not_found")
    with pytest.raises(GatewayError, match="model_not_found"):
        loader.load(models[1].version, identity)
    assert not any(key[0] == models[1].version for key in loader._cache)


@pytest.mark.parametrize("change", [{"state": "revoked"}, {"artifact_digest": "synthetic-new"}])
def test_change_during_load_never_enters_cache(
    settings: Settings,
    keyring: Keyring,
    identity: Identity,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, str],
) -> None:
    model = manifest()
    registry = Mock(spec=ModelRegistry)
    registry.get.side_effect = [model, model.model_copy(update=change)]

    class Provider(FakeProvider):
        def __init__(self, model: ModelManifest, *args: object) -> None:
            super().__init__()
            self.model_version = model.artifact_digest

    monkeypatch.setattr("adaptive_llm.routing.shadow.SpecialistProvider", Provider)
    loader = RegistrySpecialists(
        registry,
        FoundationRouter(settings.routing_path).deployment,
        keyring,
        tmp_path,
        frozenset({"synthetic-a"}),
    )
    with pytest.raises(GatewayError, match="shadow_model_ineligible"):
        loader.load(model.version, identity)
    assert not loader._cache
