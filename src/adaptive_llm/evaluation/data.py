"""Authenticated immutable held-out shards and explicitly synthetic suite fixtures."""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from adaptive_llm.contracts import (
    DatasetManifest,
    EvaluationSpecification,
    InferenceRequest,
    Message,
    RagOptions,
    ResponseFormat,
    uid,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.datasets.builder import DatasetBuilder
from adaptive_llm.evaluation.runner import Case
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.rag import IndexedChunk
from adaptive_llm.storage.crypto import PayloadCipher


class DatasetReader(Protocol):
    def read(
        self, spec: EvaluationSpecification, identity: Identity
    ) -> tuple[DatasetManifest, list[Case]]: ...


class Inputs(BaseModel):
    messages: list[Message]
    sources: list[str]


class Source(BaseModel):
    document_id: str
    chunk_id: str
    document_version: str
    document_family: str
    index_id: str
    index_version: str


class HeldOutRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tenant_id: str
    split: str
    example_hash: str
    input: Inputs
    sources: list[Source]
    target: str
    labels: dict[str, str]


def segment(keyring: Keyring, key: str, value: str) -> str:
    # Label values can originate in datasets: do not echo them into broadly readable reports.
    return f"{key}.{keyring.fingerprint(value)[:16]}"


class LocalDatasetReader:
    def __init__(
        self, builder: DatasetBuilder, data_dir: Path, cipher: PayloadCipher, keyring: Keyring
    ) -> None:
        self.builder, self.data_dir, self.cipher, self.keyring = builder, data_dir, cipher, keyring

    def read(
        self, spec: EvaluationSpecification, identity: Identity
    ) -> tuple[DatasetManifest, list[Case]]:
        manifest = self.builder.get(spec.dataset_id, spec.dataset_version, identity)
        rows = read_shards(manifest, self.data_dir, self.cipher, self.keyring)
        try:
            cases = [self._case(HeldOutRow.model_validate_json(row), spec) for row in rows["test"]]
            if len({case.item_id for case in cases}) != len(cases):
                raise ValueError
            return manifest, cases
        except Exception:
            raise GatewayError(409, "invalid_dataset_artifact") from None

    def _case(self, row: HeldOutRow, spec: EvaluationSpecification) -> Case:
        chunks: list[IndexedChunk] = []
        for source, wrapped in zip(row.sources, row.input.sources, strict=True):
            prefix = f"<<source {source.document_id}/{source.chunk_id} {source.document_version}>>"
            suffix = "<</source>>"
            if not wrapped.startswith(prefix) or not wrapped.endswith(suffix):
                raise ValueError
            chunks.append(
                IndexedChunk(
                    tenant_id=row.tenant_id,
                    allowed_applications=["evaluation"],
                    index_id=source.index_id,
                    index_version=source.index_version,
                    document_id=source.document_id,
                    document_version=source.document_version,
                    document_family=source.document_family,
                    chunk_id=source.chunk_id,
                    content=wrapped[len(prefix) : -len(suffix)],
                    licence_class="synthetic",
                )
            )
        if len({c.index_id for c in chunks}) > 1:
            raise ValueError
        labels = dict(row.labels)
        labels["citation"] = "required" if chunks else "none"
        labels["safety"] = labels.get("risk_tier", "unknown")
        return Case(
            item_id=row.example_hash,
            tenant_id=row.tenant_id,
            request=InferenceRequest(
                request_id=uid(),
                application_id="evaluation",
                messages=row.input.messages,
                rag=RagOptions(
                    enabled=bool(chunks), index_id=chunks[0].index_id if chunks else None
                ),
            ),
            corpus=tuple(chunks),
            target=row.target,
            expected_citations=frozenset((s.document_id, s.chunk_id) for s in row.sources),
            segments=tuple(
                segment(self.keyring, key, labels[key])
                for key in spec.critical_segments
                if key in labels
            ),
        )


class SyntheticItem(BaseModel):
    input: str
    target: str = ""
    expect_contains: list[str] = Field(default_factory=list)
    expect_citation: bool = True
    expect_json: bool = False
    prohibited: list[str] = Field(default_factory=list)
    category: str = "general"
    critical: bool = False
    retrieved_content: str | None = None
    k: int = 3


def fixture_cases(path: Path, tenant: str) -> list[Case]:
    cases: list[Case] = []
    for index, line in enumerate(path.read_text().splitlines()):
        item = SyntheticItem.model_validate_json(line)
        document = f"synthetic-evaluation-{index}"
        chunk = IndexedChunk(
            tenant_id=tenant,
            allowed_applications=["evaluation"],
            index_id="synthetic-evaluation",
            index_version="synthetic-evaluation-1",
            document_id=document,
            document_version="synthetic-1",
            document_family=document,
            chunk_id="chunk",
            content=item.retrieved_content or item.target,
            licence_class="synthetic",
        )
        corpus: tuple[IndexedChunk, ...] = (chunk,) if chunk.content else ()
        if item.category in {"cross_tenant", "acl", "stale"}:
            # Higher-overlap decoys must be filtered before scoring/ranking.
            decoys = (
                chunk.model_copy(
                    update={
                        "tenant_id": "synthetic-forbidden",
                        "document_id": "synthetic-foreign",
                        "content": item.input,
                    }
                ),
                chunk.model_copy(
                    update={
                        "allowed_applications": ["private"],
                        "document_id": "synthetic-private",
                        "content": item.input,
                    }
                ),
                chunk.model_copy(
                    update={
                        "index_id": "synthetic-stale-index",
                        "document_id": "synthetic-stale",
                        "content": item.input,
                    }
                ),
                chunk.model_copy(
                    update={
                        "environment": "production",
                        "document_id": "synthetic-production",
                        "content": item.input,
                    }
                ),
            )
            corpus += decoys
        cases.append(
            Case(
                item_id=f"synthetic-{path.parent.name}-{index}",
                tenant_id=tenant,
                request=InferenceRequest(
                    request_id=uid(),
                    application_id="evaluation",
                    messages=[Message(role="user", content=item.input)],
                    rag=RagOptions(
                        enabled=bool(corpus), index_id="synthetic-evaluation" if corpus else None
                    ),
                    response_format=ResponseFormat(
                        type="json_object" if item.expect_json else "text"
                    ),
                ),
                corpus=corpus,
                target=item.target,
                expected_facts=tuple(item.expect_contains),
                expected_citations=frozenset({(document, "chunk")})
                if item.expect_citation
                else frozenset(),
                prohibited=tuple(item.prohibited),
                expect_json=item.expect_json,
                critical=item.critical,
                category=item.category,
                k=item.k,
            )
        )
    return cases
