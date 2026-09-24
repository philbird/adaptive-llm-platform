"""Authenticated fake or real specialist; optional tensor work stays off the event loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from adaptive_llm.contracts import ModelManifest
from adaptive_llm.gateway.identity import Keyring
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.registry.artifacts import verify_artifact

if TYPE_CHECKING:
    from adaptive_llm.training.lora import LoraGenerator


class SpecialistProvider:
    def __init__(
        self, manifest: ModelManifest, path: Path, keyring: Keyring, data_dir: Path
    ) -> None:
        verified = verify_artifact(manifest, path, keyring)
        self.model_version = f"{manifest.base_model_revision}.{manifest.artifact_digest}"
        self.artifact_digest = manifest.artifact_digest
        self._real: LoraGenerator | None = None
        if manifest.adapter_architecture in {"lora-peft-v1", "student-full-v1"}:
            from adaptive_llm.training.lora import LoraGenerator

            self._real = LoraGenerator(manifest, data_dir, verified)
        else:
            self._weights = verified["adapter_weights.bin"]
        self._foundation = FakeProvider()

    async def soft_targets(self, row: bytes) -> bytes | None:
        if self._real is None:
            return None
        return await asyncio.to_thread(self._real.soft_targets, row)

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        if self._real is not None:
            return await asyncio.to_thread(self._real.generate, request)
        return await self._foundation.generate(request)
