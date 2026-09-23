import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from adaptive_llm.app import Settings
from adaptive_llm.contracts import RagOptions
from adaptive_llm.gateway.identity import Identity, Keyring
from adaptive_llm.rag import LocalRetriever


async def test_tenant_filter_precedes_scoring(
    settings: Settings, identity: Identity, keyring: Keyring
) -> None:
    class InspectScoring(LocalRetriever):
        @staticmethod
        def _score(query_terms: set[str], content: str) -> float:
            assert "tenant B" not in content
            return LocalRetriever._score(query_terms, content)

    query = "SYNTHETIC receipt"
    result = await InspectScoring(settings.documents_path, keyring).retrieve(
        query,
        identity,
        "support-assistant",
        RagOptions(enabled=True, index_id="synthetic-kb"),
        "interaction",
        residency="local",
    )
    assert [c.chunk_id for c in result.supplied_chunks] == ["refund-window"]
    assert [c.chunk_id for c in result.run.candidates] == ["refund-window"]
    evidence = result.run.candidates[0]
    assert evidence.document_version == "synthetic-1"
    assert result.run.index_version == "synthetic-index-1"
    assert (
        evidence.content_hash
        == hashlib.sha256(result.supplied_chunks[0].content.encode()).hexdigest()
    )
    assert evidence.context_position == 0
    assert result.run.query_hash == keyring.content_hash(query, purpose="query")
    assert result.run.query_hash != keyring.content_hash(query, purpose="input")
    assert result.run.query_hash_scheme == "hmac-sha256"
    assert result.run.query_ref is None


@pytest.mark.parametrize("constraint", ["environment", "application", "index", "acl", "region"])
async def test_access_constraints(
    documents_path: Path, identity: Identity, keyring: Keyring, constraint: str
) -> None:
    options = RagOptions(enabled=True, index_id="synthetic-kb")
    application = "support-assistant"
    if constraint == "environment":
        identity = replace(identity, environment="production")
    elif constraint == "application":
        application = "untrusted"
    elif constraint == "index":
        options = RagOptions(enabled=True, index_id="another-index")
    else:
        rows = json.loads(await asyncio.to_thread(documents_path.read_text))
        if constraint == "region":
            rows[0]["region"] = "eu-west"
        else:
            rows[0]["allowed_applications"] = ["different-application"]
        await asyncio.to_thread(documents_path.write_text, json.dumps(rows))

    class MustNotScore(LocalRetriever):
        @staticmethod
        def _score(query_terms: set[str], content: str) -> float:
            pytest.fail("unauthorised chunk reached scoring")

    result = await MustNotScore(documents_path, keyring).retrieve(
        "SYNTHETIC", identity, application, options, "interaction", residency="local"
    )
    assert not result.supplied_chunks
    assert not result.run.candidates
    assert result.run.filters["region"] == "local"


async def test_disabled_retrieval_cannot_create_a_run(
    settings: Settings, identity: Identity, keyring: Keyring
) -> None:
    with pytest.raises(ValueError, match="^retrieval_not_enabled$"):
        await LocalRetriever(settings.documents_path, keyring).retrieve(
            "SYNTHETIC",
            identity,
            "support-assistant",
            RagOptions(),
            "interaction",
            residency="local",
        )


async def test_retrieved_and_supplied_evidence_are_distinct(
    documents_path: Path, identity: Identity, keyring: Keyring
) -> None:
    rows = json.loads(await asyncio.to_thread(documents_path.read_text))
    second = {**rows[0], "chunk_id": "second", "document_version": "synthetic-2"}
    rows.append(second)
    await asyncio.to_thread(documents_path.write_text, json.dumps(rows))
    result = await LocalRetriever(documents_path, keyring, supplied_limit=1).retrieve(
        "SYNTHETIC receipt",
        identity,
        "support-assistant",
        RagOptions(enabled=True, index_id="synthetic-kb"),
        "interaction",
        residency="local",
    )
    assert len(result.run.candidates) == 2
    assert [c.rank_retrieved for c in result.run.candidates] == [1, 2]
    assert [c.supplied_to_model for c in result.run.candidates] == [True, False]
    assert result.run.candidates[1].context_position is None
    assert len(result.supplied_chunks) == 1
    bounded = await LocalRetriever(documents_path, keyring, context_token_budget=1).retrieve(
        "SYNTHETIC",
        identity,
        "support-assistant",
        RagOptions(enabled=True, index_id="synthetic-kb"),
        "interaction",
        residency="local",
    )
    assert len(bounded.run.candidates) == 2
    assert not bounded.supplied_chunks


def test_mixed_versions_are_rejected(documents_path: Path, keyring: Keyring) -> None:
    rows = json.loads(documents_path.read_text())
    rows.append({**rows[0], "chunk_id": "second", "index_version": "inconsistent"})
    documents_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="^mixed_index_versions$"):
        LocalRetriever(documents_path, keyring)
