"""OpenRouter chat completion adapter. No transport diagnostics cross this boundary."""

import asyncio
import logging
import re
from contextvars import ContextVar
from time import perf_counter
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from adaptive_llm.contracts import Citation, Usage
from adaptive_llm.providers import (
    CITATION_PATTERN,
    ProviderError,
    ProviderRequest,
    ProviderResult,
    render_context,
)

BASE_URL = "https://openrouter.ai/api/v1"
_private_transport: ContextVar[bool] = ContextVar("openrouter_private_transport", default=False)


class TransportLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _private_transport.get()


# httpcore debug records can include upstream response headers. Suppress only this
# adapter's task-local transport records, without muting concurrent gateway traffic.
for _logger_name in (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
    "httpcore.connection_pool",
):
    logging.getLogger(_logger_name).addFilter(TransportLogFilter())


def outbound_request(model: str, request: ProviderRequest) -> dict[str, object]:
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if request.context:
        messages.append({"role": "user", "content": render_context(request.context)})
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "provider": {"data_collection": "deny"},
    }
    schema = request.response_format.json_schema
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.name, "schema": schema.schema_, "strict": True},
        }
    elif request.response_format.type == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def strip_json_fence(content: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n?```\s*", content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else content


class TokenDetails(BaseModel):
    cached_tokens: int | None = Field(default=None, ge=0, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)


class ReportedUsage(BaseModel):
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    prompt_tokens_details: TokenDetails | None = None
    completion_tokens_details: TokenDetails | None = None


class CompletionMessage(BaseModel):
    model_config = ConfigDict(strict=True)
    content: str = Field(max_length=32000, repr=False)
    tool_calls: list[object] | None = None


class Choice(BaseModel):
    message: CompletionMessage
    finish_reason: str | None = Field(strict=True)


class Completion(BaseModel):
    choices: list[Choice] = Field(min_length=1, max_length=1)
    usage: ReportedUsage


class OpenRouterProvider:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("openrouter_api_key_required")
        self.model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._transport = transport
        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

    def _client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = httpx.AsyncClient(
                transport=self._transport, trust_env=False, follow_redirects=False
            )
        return self._clients[loop]

    async def aclose(self) -> None:
        """Release this loop's pool; other serving/evaluation loops own their pools."""
        client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            token = _private_transport.set(True)
            try:
                await client.aclose()
            finally:
                _private_transport.reset(token)

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        started = perf_counter()
        token = _private_transport.set(True)
        try:
            async with asyncio.timeout(request.deadline_ms / 1000):
                response = await self._client().post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "HTTP-Referer": "https://github.com/philbird/adaptive-llm-platform",
                        "X-Title": "Adaptive LLM Specialisation Platform",
                    },
                    json=outbound_request(self.model, request),
                    timeout=request.deadline_ms / 1000,
                )
            if response.status_code == 429:
                raise ProviderError("provider_rate_limited")
            if response.status_code in {408, 504}:
                raise ProviderError("provider_timeout")
            if response.status_code != 200:
                raise ProviderError("provider_unavailable")
            completion = Completion.model_validate_json(response.content)
            choice, usage = completion.choices[0], completion.usage
            if choice.message.tool_calls or (
                choice.finish_reason
                not in {
                    "stop",
                    "length",
                    "content_filter",
                    "end_turn",
                    "max_tokens",
                    "stop_sequence",
                }
                and not choice.message.content.strip()
            ):
                raise ProviderError("provider_invalid_response")
            content = choice.message.content
            if request.response_format.type != "text":
                content = strip_json_fence(content)
            finish: Literal["stop", "length", "content_filter"] = (
                "length"
                if choice.finish_reason in {"length", "max_tokens"}
                else "content_filter"
                if choice.finish_reason == "content_filter"
                else "stop"
            )
            return ProviderResult(
                content=content,
                citations=tuple(
                    Citation(document_id=document, chunk_id=chunk)
                    for document, chunk in dict.fromkeys(CITATION_PATTERN.findall(content))
                ),
                usage=Usage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cached_input_tokens=usage.prompt_tokens_details.cached_tokens
                    if usage.prompt_tokens_details
                    else None,
                    reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                    if usage.completion_tokens_details
                    else None,
                    source="provider_reported",
                    tokenizer=self.model,
                ),
                finish_reason=finish,
                latency_ms=(perf_counter() - started) * 1000,
            )
        except ProviderError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise ProviderError("provider_timeout") from None
        except httpx.RequestError:
            raise ProviderError("provider_unavailable") from None
        except Exception:
            raise ProviderError("provider_invalid_response") from None
        finally:
            _private_transport.reset(token)
