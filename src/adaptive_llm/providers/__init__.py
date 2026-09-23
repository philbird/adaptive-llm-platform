"""Canonical provider boundary and an entirely local deterministic foundation provider."""

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from adaptive_llm.contracts import Chunk, Citation, FinishReason, Message, ResponseFormat, Usage

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


class Provider(Protocol):
    async def generate(self, request: ProviderRequest) -> ProviderResult: ...


class FakeProvider:
    """Whitespace tokens, no I/O and zero simulated latency; controls are injection-only."""

    def __init__(
        self, *, test_only_failure: Literal["error", "deadline_exceeded"] | None = None
    ) -> None:
        self._failure = test_only_failure

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        content = "SYNTHETIC ANSWER: " + (
            " ".join(f"{c.content} [{c.document_id}/{c.chunk_id}]" for c in request.context)
            if request.context
            else "No context supplied."
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
