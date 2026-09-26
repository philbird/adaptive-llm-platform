"""Canonical provider boundary and an entirely local deterministic foundation provider."""

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from adaptive_llm.contracts import (
    Chunk,
    Citation,
    FinishReason,
    Message,
    ResponseFormat,
    ToolCall,
    Usage,
)

TOKENIZER = "fake-whitespace-v1"
CITATION_PATTERN = re.compile(r"\[([\w.-]+)/([\w.-]+)\]")


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


@dataclass(frozen=True)
class ProviderRequest:
    messages: tuple[Message, ...]
    context: tuple[Chunk, ...]
    response_format: ResponseFormat
    max_output_tokens: int
    application_id: str = ""
    deadline_ms: float = 5000
    temperature: float = 0

    @property
    def input_tokens(self) -> int:
        return sum(token_count(m.content) for m in self.messages) + sum(
            token_count(c.content) for c in self.context
        )


@dataclass(frozen=True)
class ProviderResult:
    content: str
    citations: tuple[Citation, ...]
    usage: Usage
    finish_reason: FinishReason
    latency_ms: float
    tool_calls: tuple[ToolCall, ...] = ()


class Provider(Protocol):
    async def generate(self, request: ProviderRequest) -> ProviderResult: ...


@runtime_checkable
class LoopScopedProvider(Protocol):
    async def aclose(self) -> None:
        """Close resources owned by the calling event loop, after its requests finish."""
        ...


ProviderErrorCode = Literal[
    "provider_timeout", "provider_rate_limited", "provider_unavailable", "provider_invalid_response"
]


class ProviderError(Exception):
    def __init__(self, code: ProviderErrorCode) -> None:
        super().__init__(code)
        self.code = code


def render_context(context: tuple[Chunk, ...]) -> str:
    return " ".join(f"{c.content} [{c.document_id}/{c.chunk_id}]" for c in context)


@runtime_checkable
class SoftTargetProvider(Protocol):
    async def soft_targets(self, row: bytes) -> bytes | None:
        """Optional target-token log distributions, serialized exclusively as safetensors."""
        ...


class FakeProvider:
    """Whitespace tokens, no I/O and zero simulated latency; controls are injection-only."""

    def __init__(
        self, *, test_only_failure: Literal["error", "deadline_exceeded"] | None = None
    ) -> None:
        self._failure = test_only_failure

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        content = "SYNTHETIC ANSWER: " + (
            render_context(request.context) if request.context else "No context supplied."
        )
        if request.response_format.type == "json_object":
            content = json.dumps({"answer": content}, ensure_ascii=False)
        tokens = list(re.finditer(r"\S+", content))
        finish: FinishReason = "stop"
        if len(tokens) > request.max_output_tokens:
            content = content[: tokens[request.max_output_tokens - 1].end()]
            finish = "length"
        if self._failure is not None:
            content = ""
            finish = self._failure
        citations = tuple(
            Citation(document_id=c.document_id, chunk_id=c.chunk_id)
            for c in request.context
            if f"[{c.document_id}/{c.chunk_id}]" in content
        )
        return ProviderResult(
            content=content,
            citations=citations,
            usage=Usage(
                input_tokens=request.input_tokens,
                output_tokens=token_count(content),
                source="locally_estimated",
                tokenizer=TOKENIZER,
            ),
            finish_reason=finish,
            latency_ms=0.0,
        )
