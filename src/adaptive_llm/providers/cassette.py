"""Explicit fixture-only recording; replay can never invoke its delegate."""

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adaptive_llm.contracts import Citation, Usage
from adaptive_llm.providers import (
    CITATION_PATTERN,
    LoopScopedProvider,
    Provider,
    ProviderRequest,
    ProviderResult,
)
from adaptive_llm.providers.openrouter import OpenRouterProvider, outbound_request
from adaptive_llm.structured import canonical

DIRECTORY = Path(__file__).resolve().parents[3] / "tests/fixtures/cassettes/openrouter"


def request_hash(model: str, request: ProviderRequest) -> str:
    return hashlib.sha256(canonical(outbound_request(model, request)).encode("utf-8")).hexdigest()


class CassetteMissing(AssertionError):
    def __init__(self, digest: str) -> None:
        super().__init__(f"openrouter cassette missing: {digest}")


class CanonicalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    content: str = Field(max_length=32000, repr=False)
    usage: Usage
    finish_reason: Literal["stop", "length", "content_filter"]
    latency_ms: float = Field(ge=0)

    @field_validator("usage", mode="before")
    @classmethod
    def canonical_usage_only(cls, value: object) -> object:
        if isinstance(value, dict) and not value.keys() <= Usage.model_fields.keys():
            raise ValueError("invalid_cassette_usage")
        return value

    def result(self) -> ProviderResult:
        return ProviderResult(
            content=self.content,
            usage=self.usage,
            finish_reason=self.finish_reason,
            latency_ms=self.latency_ms,
            citations=tuple(
                Citation(document_id=document, chunk_id=chunk)
                for document, chunk in dict.fromkeys(CITATION_PATTERN.findall(self.content))
            ),
        )


class CassetteProvider:
    def __init__(
        self,
        model: str,
        directory: Path = DIRECTORY,
        *,
        mode: Literal["replay", "record"] = "replay",
        delegate: Provider | None = None,
    ) -> None:
        if mode not in {"replay", "record"}:
            raise ValueError("invalid_cassette_mode")
        if mode == "record" and (
            os.environ.get("OPENROUTER_LIVE") != "1"
            or not os.environ.get("OPENROUTER_API_KEY", "").strip()
            or delegate is None
        ):
            raise ValueError("openrouter_recording_not_authorised")
        self.model, self.directory, self.mode, self.delegate = model, directory, mode, delegate

    def require(self, request: ProviderRequest) -> None:
        digest = request_hash(self.model, request)
        if not (self.directory / f"{digest}.json").is_file():
            raise CassetteMissing(digest)

    async def aclose(self) -> None:
        if self.mode == "record" and isinstance(self.delegate, LoopScopedProvider):
            await self.delegate.aclose()

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        digest = request_hash(self.model, request)
        path = self.directory / f"{digest}.json"
        if path.is_file():
            try:
                raw = await asyncio.to_thread(path.read_text)
                return CanonicalResult.model_validate_json(raw).result()
            except Exception:
                raise ValueError("invalid_openrouter_cassette") from None
        if self.mode == "replay":
            raise CassetteMissing(digest)
        assert self.delegate is not None
        result = await self.delegate.generate(request)
        if result.tool_calls:
            raise ValueError("invalid_openrouter_cassette")
        try:
            record = CanonicalResult.model_validate(
                {
                    "content": result.content,
                    "usage": result.usage,
                    "finish_reason": result.finish_reason,
                    "latency_ms": result.latency_ms,
                }
            )
        except Exception:
            raise ValueError("invalid_openrouter_cassette") from None
        await asyncio.to_thread(self._append, path, record.model_dump_json(indent=2))
        return record.result()

    @staticmethod
    def _append(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as stream:
                stream.write(content + "\n")
        except FileExistsError:
            pass


async def record_fixtures(directory: Path) -> None:
    from adaptive_llm.evaluation.data import fixture_cases
    from adaptive_llm.policy import ProcessingRedactor

    model = "anthropic/claude-haiku-4.5"
    provider = CassetteProvider(
        model,
        directory,
        mode="record",
        delegate=OpenRouterProvider(model, os.environ.get("OPENROUTER_API_KEY", "")),
    )
    try:
        root = DIRECTORY.parents[1]
        for suite in ("golden", "safety"):
            cases = fixture_cases(
                root / suite / "manyfails-triage.jsonl", "manyfails", "research-sweep"
            )
            for case in cases:
                request, _ = ProcessingRedactor().redact(case.request)
                canonical_request = ProviderRequest(
                    messages=tuple(request.messages),
                    context=(),
                    response_format=request.response_format,
                    max_output_tokens=request.max_output_tokens,
                    deadline_ms=request.routing.deadline_ms,
                    application_id=request.application_id,
                )
                await provider.generate(canonical_request)
                print(request_hash(model, canonical_request))
    finally:
        await provider.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record approved ManyFails fixture cassettes")
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    try:
        asyncio.run(record_fixtures(args.directory))
    except Exception:
        parser.exit(1, "openrouter_recording_failed\n")


if __name__ == "__main__":
    main()
