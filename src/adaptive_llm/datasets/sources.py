"""Replaceable exact-version source resolver; tenant/ACL checks precede content access."""

import hashlib
import json
from pathlib import Path
from typing import Protocol

from adaptive_llm.contracts import Chunk, ChunkEvidence, Interaction, RetrievalRun
from adaptive_llm.rag import IndexedChunk


class SourceResolver(Protocol):
    def resolve(
        self,
        interaction: Interaction,
        retrieval: RetrievalRun,
        evidence: ChunkEvidence,
        residency: str,
    ) -> Chunk | None: ...


class LocalSourceResolver:
    def __init__(self, path: Path) -> None:
        self._path = path

    def resolve(
        self,
        interaction: Interaction,
        retrieval: RetrievalRun,
        evidence: ChunkEvidence,
        residency: str,
    ) -> Chunk | None:
        # Read only during dataset construction, so an injected serving retriever does not
        # acquire a new fixture-file dependency. Rebuilds observe removed/relicensed sources.
        try:
            chunks = [
                IndexedChunk.model_validate(row) for row in json.loads(self._path.read_text())
            ]
        except (OSError, ValueError, TypeError):
            return None
        matches = [
            chunk
            for chunk in chunks
            if chunk.tenant_id == interaction.tenant_id
            and chunk.environment == interaction.environment
            and interaction.application_id in chunk.allowed_applications
            and chunk.region == residency
            and chunk.index_id == retrieval.index_id
            and chunk.index_version == retrieval.index_version
            and chunk.document_id == evidence.document_id
            and chunk.chunk_id == evidence.chunk_id
            and chunk.document_version == evidence.document_version
            and chunk.licence_class in {"synthetic", "internal-approved"}
            and chunk.licence_class == evidence.licence_class
        ]
        if len(matches) != 1:
            return None
        chunk = matches[0]
        if hashlib.sha256(chunk.content.encode()).hexdigest() != evidence.content_hash:
            return None
        return chunk
