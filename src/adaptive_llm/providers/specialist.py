"""The fake specialist loads authenticated artifacts and retains foundation behavior."""

from pathlib import Path

from adaptive_llm.contracts import ModelManifest
from adaptive_llm.gateway.identity import Keyring
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.registry.artifacts import verify_artifact


class SpecialistProvider:
    def __init__(self, manifest: ModelManifest, path: Path, keyring: Keyring) -> None:
        verified = verify_artifact(manifest, path, keyring)
        self.model_version = f"{manifest.base_model_revision}.{manifest.artifact_digest}"
        self.artifact_digest = manifest.artifact_digest
        self._weights = verified["adapter_weights.bin"]
        self._foundation = FakeProvider()

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        return await self._foundation.generate(request)
