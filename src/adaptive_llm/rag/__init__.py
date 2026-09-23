"""ACL-first lexical retrieval with exact source evidence and bounded context."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from adaptive_llm.contracts import Chunk, ChunkEvidence, RagOptions, Region, RetrievalRun
from adaptive_llm.gateway.identity import Identity, Keyring
from adaptive_llm.providers import token_count


@dataclass(frozen=True)
class RetrievalResult:
    run: RetrievalRun
    supplied_chunks: tuple[Chunk, ...]


class Retriever(Protocol):
    async def retrieve(
        self,
        query: str,
        identity: Identity,
        application_id: str,
        options: RagOptions,
        interaction_id: str,
        *,
        residency: Region,
    ) -> RetrievalResult: ...


class IndexedChunk(Chunk):
    index_version: str


class LocalRetriever:
    def __init__(
        self,
        path: Path,
        keyring: Keyring,
        *,
        candidate_limit: int = 10,
        supplied_limit: int = 3,
        context_token_budget: int = 2048,
    ) -> None:
        if min(candidate_limit, supplied_limit, context_token_budget) < 1:
            raise ValueError("invalid_retrieval_limits")
        self._chunks = tuple(
            IndexedChunk.model_validate(row) for row in json.loads(path.read_text())
        )
        self._versions: dict[tuple[str, str, str, str], str] = {}
        for chunk in self._chunks:
            key = (chunk.tenant_id, chunk.environment, chunk.region, chunk.index_id)
            if self._versions.setdefault(key, chunk.index_version) != chunk.index_version:
                raise ValueError("mixed_index_versions")
        self._keyring = keyring
        self._candidate_limit = candidate_limit
        self._supplied_limit = supplied_limit
        self._context_token_budget = context_token_budget

    @staticmethod
    def _score(query_terms: set[str], content: str) -> float:
        terms = set(re.findall(r"\w+", content.lower()))
        return len(query_terms & terms) / max(len(query_terms), 1)

    async def retrieve(
        self,
        query: str,
        identity: Identity,
        application_id: str,
        options: RagOptions,
        interaction_id: str,
        *,
        residency: Region,
    ) -> RetrievalResult:
        started = perf_counter()
        if not options.enabled or options.index_id is None:
            raise ValueError("retrieval_not_enabled")
        # Never score a chunk until every access constraint has passed.
        eligible = [
            chunk
            for chunk in self._chunks
            if application_id in identity.application_ids
            and chunk.tenant_id == identity.tenant_id
            and chunk.environment == identity.environment
            and chunk.region == residency
            and application_id in chunk.allowed_applications
            and chunk.index_id == options.index_id
        ]
        terms = set(re.findall(r"\w+", query.lower()))
        scored = [(self._score(terms, c.content), c) for c in eligible]
        ranked = sorted(
            ((score, chunk) for score, chunk in scored if score > 0),
            key=lambda pair: (-pair[0], pair[1].document_id, pair[1].chunk_id),
        )[: self._candidate_limit]
        supplied: list[Chunk] = []
        evidence: list[ChunkEvidence] = []
        remaining = self._context_token_budget
        for rank, (score, chunk) in enumerate(ranked, start=1):
            count = token_count(chunk.content)
            include = len(supplied) < self._supplied_limit and count <= remaining
            if include:
                supplied.append(chunk)
                remaining -= count
            evidence.append(
                ChunkEvidence(
                    document_id=chunk.document_id,
                    document_version=chunk.document_version,
                    chunk_id=chunk.chunk_id,
                    rank_retrieved=rank,
                    retrieval_score=score,
                    supplied_to_model=include,
                    context_position=len(supplied) - 1 if include else None,
                    token_count=count,
                    content_hash=hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                    licence_class=chunk.licence_class,
                )
            )
        index_id = options.index_id
        return RetrievalResult(
            run=RetrievalRun(
                interaction_id=interaction_id,
                index_id=index_id,
                index_version=self._versions.get(
                    (identity.tenant_id, identity.environment, residency, index_id),
                    "unavailable",
                ),
                filters={
                    "tenant_id": identity.tenant_id,
                    "environment": identity.environment,
                    "region": residency,
                    "application_id": application_id,
                    "index_id": index_id,
                },
                query_hash=self._keyring.content_hash(query, purpose="query"),
                latency_ms=(perf_counter() - started) * 1000,
                candidates=evidence,
            ),
            supplied_chunks=tuple(supplied),
        )
