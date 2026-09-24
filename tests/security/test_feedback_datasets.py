import base64
import json
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from postgres_support import stored_bytes

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Feedback, Interaction, now
from adaptive_llm.storage import EncryptedPayload, StorageError

USER = {"Authorization": "Bearer synthetic-key-a", "X-Subject": "synthetic-label-actor"}
OTHER = {"Authorization": "Bearer synthetic-key-b"}
OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}

if TYPE_CHECKING:
    from conftest import DatasetSeed


@pytest.mark.parametrize("correction", [False, True])
def test_feedback_isolation_encrypted_redacted_refs_and_tombstones(
    dataset_seed: "DatasetSeed",
    correction: bool,
) -> None:
    seed = dataset_seed
    iid = seed.ids[3]
    suffix = "correction" if correction else "feedback"
    text = "SYNTHETIC label actor@example.test sk-syntheticSecret1234"
    body = (
        {"correction": text, "training_authorised": True}
        if correction
        else {
            "label_type": "rubric",
            "value": {"score": 4, "max_score": 5},
            "comment": text,
            "training_authorised": True,
        }
    )
    url = f"/v1/interactions/{iid}/{suffix}"
    assert seed.client.post(url, headers=OTHER, json=body).status_code == 404
    assert seed.client.post(url, json=body).status_code == 401
    if correction:
        assert seed.client.post(url, headers=USER, json={"correction": text}).status_code == 422
        assert (
            seed.client.post(
                url, headers=USER, json={"correction": text, "training_authorised": "yes"}
            ).status_code
            == 422
        )
        assert (
            seed.client.post(
                url, headers=USER, json={"correction": "x" * 32001, "training_authorised": True}
            ).status_code
            == 422
        )
    response = seed.client.post(url, headers=USER, json=body)
    assert response.status_code == 200
    feedback = Feedback.model_validate(response.json())
    assert feedback.source == "user"
    assert feedback.training_authorised
    assert feedback.actor_id_pseudonymous == seed.app.state.keyring.pseudonym(
        "synthetic-a", USER["X-Subject"]
    )
    field = "correction" if correction else "comment"
    ref = getattr(feedback, f"{field}_ref")
    blob = seed.app.state.payloads.get("synthetic-a", ref, now())
    assert blob.field == field
    plain = seed.app.state.persistence.cipher.decrypt(blob, "synthetic-a", iid, field).decode()
    assert plain == "SYNTHETIC label [REDACTED] [REDACTED]"
    assert seed.app.state.payloads.get("synthetic-b", ref, now()) is None
    interaction = seed.app.state.metadata.get("synthetic-a", Interaction, iid)
    assert interaction.feedback_ids[-1] == feedback.feedback_id
    while seed.app.state.dispatcher.dispatch_once():
        pass
    events = seed.app.state.events.events_for_trace(interaction.trace_id)
    assert events[-1].event_type == "feedback.recorded.v1"
    assert events[-1].data == feedback
    assert "actor@example.test" not in events[-1].model_dump_json()
    assert text.encode() not in stored_bytes(seed.app.state.database)
    assert seed.client.delete(f"/v1/privacy/interactions/{iid}", headers=USER).status_code == 204
    assert seed.client.post(url, headers=USER, json=body).status_code == 409
    assert seed.app.state.payloads.get("synthetic-a", ref, now()) is None
    stored = seed.app.state.metadata.get("synthetic-a", Feedback, feedback.feedback_id)
    assert getattr(stored, f"{field}_ref") is None


def test_logging_disabled_and_redaction_failure_leave_no_feedback_content(
    dataset_seed: "DatasetSeed", monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = dataset_seed
    iid = seed.ids[3]
    body = {"correction": "SYNTHETIC private correction", "training_authorised": True}
    seed.policy.logging = False
    response = seed.client.post(f"/v1/interactions/{iid}/correction", headers=USER, json=body)
    assert response.status_code == 200 and response.json()["correction_ref"] is None
    assert response.json()["content_hash"] is None
    seed.policy.logging = True

    def fail(*args: object) -> None:
        raise RuntimeError("SYNTHETIC sensitive failure")

    monkeypatch.setattr(seed.app.state.persistence.redactor, "redact_text", fail)
    response = seed.client.post(f"/v1/interactions/{iid}/correction", headers=USER, json=body)
    assert response.status_code == 200 and response.json()["correction_ref"] is None
    assert response.json()["error_code"] == "persistence_redaction_failed"


def test_feedback_transaction_rolls_back_payload_record_link_and_event(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed
    iid = seed.ids[3]
    before = seed.app.state.metadata.get("synthetic-a", Interaction, iid)
    db = seed.app.state.database.connection
    counts = [
        db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("payloads", "feedback", "outbox")
    ]

    def fail(*args: object) -> None:
        raise RuntimeError("SYNTHETIC sensitive failure")

    monkeypatch.setattr(seed.app.state.outbox, "enqueue", fail)
    result = seed.client.post(
        f"/v1/interactions/{iid}/correction",
        headers=USER,
        json={"correction": "SYNTHETIC correction", "training_authorised": True},
    )
    assert result.status_code == 503
    assert seed.app.state.metadata.get("synthetic-a", Interaction, iid) == before
    assert counts == [
        db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("payloads", "feedback", "outbox")
    ]


def test_shard_aad_binds_tenant_version_split_and_manifest_access(
    dataset_seed: "DatasetSeed",
) -> None:
    seed = dataset_seed
    response = seed.client.post(
        "/v1/datasets/builds", headers=OPERATOR, json=seed.specification.model_dump(mode="json")
    )
    assert response.status_code == 200
    manifest = response.json()
    path = seed.directory / "datasets" / manifest["dataset_id"] / manifest["version"]
    row = json.loads((path / "synthetic-a.train.jsonl.enc").read_text())
    binding = f"{manifest['dataset_id']}/{manifest['version']}"
    blob = EncryptedPayload(
        "synthetic",
        "synthetic-a",
        binding,
        "dataset",
        base64.b64decode(row["nonce"]),
        base64.b64decode(row["ciphertext"]),
        row["key_version"],
        now() + timedelta(hours=1),
    )
    cipher = seed.app.state.persistence.cipher
    for tenant, version, split in [
        ("synthetic-b", binding, "train"),
        ("synthetic-a", binding + "x", "train"),
        ("synthetic-a", binding, "test"),
    ]:
        with pytest.raises(StorageError, match="payload_authentication_failed"):
            cipher.decrypt(blob, tenant, version, "dataset", aad_field=split)
    url = f"/v1/datasets/{manifest['dataset_id']}/versions/{manifest['version']}"
    assert seed.client.get(url, headers=USER).status_code == 403
    operator = seed.app.state.authenticator.authenticate(OPERATOR["Authorization"], None)
    with pytest.raises(Exception, match="dataset_not_found"):
        seed.app.state.datasets.get(
            manifest["dataset_id"],
            manifest["version"],
            replace(operator, dataset_tenants=frozenset({"synthetic-b"})),
        )


def test_default_policy_feedback_has_null_refs() -> None:
    app = create_app(Settings(outbox_dispatch_enabled=False))
    with TestClient(app) as client:
        iid = client.post(
            "/v1/inference",
            headers=USER,
            json={
                "request_id": "synthetic",
                "application_id": "support-assistant",
                "messages": [{"role": "user", "content": "SYNTHETIC input"}],
            },
        ).json()["interaction_id"]
        result = client.post(
            f"/v1/interactions/{iid}/feedback",
            headers=USER,
            json={
                "label_type": "thumb",
                "value": {"score": 1, "max_score": 1},
                "comment": "SYNTHETIC comment",
            },
        )
        assert result.status_code == 200 and result.json()["comment_ref"] is None
