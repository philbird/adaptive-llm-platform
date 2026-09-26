import json
from dataclasses import replace

import pytest

from adaptive_llm.contracts import Message, ResponseFormat
from adaptive_llm.providers import FakeProvider
from adaptive_llm.providers.cassette import CassetteMissing, CassetteProvider, request_hash

MODEL = "synthetic-model"


async def test_replay_miss_never_calls_provider_even_with_live_env(
    provider_request, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC-KEY")

    class Forbidden:
        async def generate(self, request):
            pytest.fail("network delegate invoked")

    provider = CassetteProvider(MODEL, tmp_path, delegate=Forbidden())
    digest = request_hash(MODEL, provider_request)
    with pytest.raises(CassetteMissing, match=f"^openrouter cassette missing: {digest}$"):
        await provider.generate(provider_request)
    assert not list(tmp_path.iterdir())


async def test_record_append_replay_and_only_canonical_result(
    provider_request, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC-KEY")
    calls = []

    class Counting(FakeProvider):
        async def generate(self, request):
            calls.append(1)
            return await super().generate(request)

    recorder = CassetteProvider(MODEL, tmp_path, mode="record", delegate=Counting())
    first = await recorder.generate(provider_request)
    path = tmp_path / f"{request_hash(MODEL, provider_request)}.json"
    original = path.read_bytes()
    assert await recorder.generate(provider_request) == first
    assert await CassetteProvider(MODEL, tmp_path).generate(provider_request) == first
    assert calls == [1] and path.read_bytes() == original
    obj = json.loads(original)
    assert set(obj) == {"content", "usage", "finish_reason", "latency_ms"}
    assert set(obj["usage"]) == {
        "schema_version",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "source",
        "tokenizer",
    }
    assert "SYNTHETIC-KEY" not in original.decode()
    assert "messages" not in obj and "headers" not in obj and "body" not in obj
    assert provider_request.messages[-1].content not in original.decode()
    path.write_text(json.dumps({**obj, "headers": {"authorization": "SYNTHETIC-KEY"}}))
    with pytest.raises(ValueError, match="^invalid_openrouter_cassette$"):
        await CassetteProvider(MODEL, tmp_path).generate(provider_request)


@pytest.mark.parametrize("live,key", [(None, None), ("0", "SYNTHETIC"), ("1", None)])
def test_record_requires_explicit_live_and_key(tmp_path, monkeypatch, live, key):
    for name, value in (("OPENROUTER_LIVE", live), ("OPENROUTER_API_KEY", key)):
        monkeypatch.delenv(name, raising=False)
        if value:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="^openrouter_recording_not_authorised$"):
        CassetteProvider(MODEL, tmp_path, mode="record", delegate=FakeProvider())


def test_hash_keys_actual_outbound_parameters_only(provider_request):
    base = request_hash(MODEL, provider_request)
    assert base == request_hash(
        MODEL, replace(provider_request, application_id="another", deadline_ms=10)
    )
    assert base != request_hash("another", provider_request)
    for changed in (
        replace(provider_request, max_output_tokens=42),
        replace(provider_request, temperature=1),
        replace(provider_request, messages=(Message(role="user", content="SYNTHETIC different"),)),
        replace(provider_request, context=()),
        replace(provider_request, response_format=ResponseFormat(type="json_object")),
    ):
        assert base != request_hash(MODEL, changed)
    a = ResponseFormat.model_validate(
        {
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "object", "required": ["x"]}},
        }
    )
    b = ResponseFormat.model_validate(
        {
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"required": ["x"], "type": "object"}},
        }
    )
    assert request_hash(MODEL, replace(provider_request, response_format=a)) == request_hash(
        MODEL, replace(provider_request, response_format=b)
    )


async def test_recorder_covers_approved_fixtures_and_prints_only_hashes(
    tmp_path, monkeypatch, capsys
):
    from adaptive_llm.providers import cassette

    monkeypatch.setenv("OPENROUTER_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC-recorder-key")
    seen = []

    class RecordingFake(FakeProvider):
        async def generate(self, request):
            seen.append(request)
            return await super().generate(request)

    monkeypatch.setattr(cassette, "OpenRouterProvider", lambda model, key: RecordingFake())
    await cassette.record_fixtures(tmp_path)
    assert len(seen) == 38
    assert all(
        r.messages[0].role == "system" and r.max_output_tokens == 400 and not r.context
        for r in seen
    )
    assert len(list(tmp_path.glob("*.json"))) == 38
    assert all(
        len(line) == 64 and int(line, 16) >= 0 for line in capsys.readouterr().out.splitlines()
    )
    await cassette.record_fixtures(tmp_path)
    assert len(seen) == 38


def test_nested_cassette_metadata_cannot_smuggle_headers():
    from pydantic import ValidationError

    from adaptive_llm.providers.cassette import CanonicalResult

    with pytest.raises(ValidationError):
        CanonicalResult.model_validate(
            {
                "content": "SYNTHETIC",
                "finish_reason": "stop",
                "latency_ms": 0,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "source": "provider_reported",
                    "tokenizer": "synthetic",
                    "headers": {"Authorization": "SYNTHETIC"},
                },
            }
        )


async def test_bad_record_result_is_content_free(provider_request, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC-recorder-key")

    class Bad(FakeProvider):
        async def generate(self, request):
            return replace(await super().generate(request), content="SYNTHETIC-private" * 4000)

    with pytest.raises(ValueError, match="^invalid_openrouter_cassette$"):
        await CassetteProvider(MODEL, tmp_path, mode="record", delegate=Bad()).generate(
            provider_request
        )
    assert not list(tmp_path.iterdir())
