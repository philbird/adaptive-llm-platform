import json
from pathlib import Path

from adaptive_llm.contracts import Chunk

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "documents.json"


def test_fixture_documents_are_synthetic_valid_chunks() -> None:
    chunks = [Chunk.model_validate(row) for row in json.loads(FIXTURES.read_text())]
    assert chunks
    assert all(c.licence_class == "synthetic" for c in chunks)
    assert all("SYNTHETIC" in c.content for c in chunks)
    assert len({c.tenant_id for c in chunks}) > 1, "fixtures must cover more than one tenant"
