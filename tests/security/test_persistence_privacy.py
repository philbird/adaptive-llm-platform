import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import (
    Feedback,
    FeedbackValue,
    GenerationAttempt,
    InferenceRequest,
    Interaction,
    PolicyDecision,
    RetrievalRun,
    RouteDecision,
    Started,
    now,
    uid,
)
from adaptive_llm.gateway.identity import Identity
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult
from adaptive_llm.storage import StorageError
from adaptive_llm.storage.retention import sweep

HEADERS = {"Authorization": "Bearer synthetic-key-a", "X-Subject": "synthetic-raw-subject"}
OTHER = {"Authorization": "Bearer synthetic-key-b", "X-Subject": "synthetic-raw-subject"}


class LoggingPolicy:
    def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
        return PolicyDecision(
            policy_version="synthetic-logging", content_logging_allowed=True, retention_seconds=60
        )


class PiiProvider(FakeProvider):
    async def generate(self, request: ProviderRequest) -> ProviderResult:
        return replace(
            await super().generate(request),
            content="SYNTHETIC OUTPUT: contact response@example.test +1 (202) 555-0101",
        )


def test_no_plaintext_in_database_and_only_redacted_encrypted_payloads(
    tmp_path: Path, inference_request: InferenceRequest, caplog: pytest.LogCaptureFixture
) -> None:
    prompt = "SYNTHETIC: receipt return window prompt@example.test +44 7700 900123"
    body = inference_request.model_dump()
    body["messages"][0]["content"] = prompt
    body["metadata"] = {"synthetic_note": "metadata@example.test"}
    app = create_app(Settings(data_dir=tmp_path, policy=LoggingPolicy(), provider=PiiProvider()))
    with TestClient(app) as client:
        result = client.post("/v1/inference", headers=HEADERS, json=body)
        client.app.state.dispatcher.dispatch_once()
        assert result.status_code == 200
        assert "response@example.test" in result.json()["content"]
        iid = result.json()["interaction_id"]
        meta, payloads, persistence = app.state.metadata, app.state.payloads, app.state.persistence
        interaction = meta.get("synthetic-a", Interaction, iid)
        retrieval = meta.get("synthetic-a", RetrievalRun, interaction.retrieval_run_id)
        attempt = meta.get("synthetic-a", GenerationAttempt, interaction.final_attempt_id)
        emitted_retrieval, emitted_attempt = app.state.events.events[1:4:2]
        assert emitted_retrieval.data.model_dump(exclude={"query_ref"}) == retrieval.model_dump(
            exclude={"query_ref"}
        )
        assert emitted_attempt.data.model_dump(exclude={"output_ref"}) == attempt.model_dump(
            exclude={"output_ref"}
        )
        assert interaction.policy.persistence_redaction_counts == {"emails": 3, "phones": 2}
        redacted_messages = None
        for field, ref in (
            ("messages", interaction.input.messages_ref),
            ("query", retrieval.query_ref),
            ("output", attempt.output_ref),
        ):
            assert ref is not None
            blob = payloads.get("synthetic-a", ref, now())
            content = persistence.cipher.decrypt(blob, "synthetic-a", iid, field).decode()
            assert "[REDACTED]" in content
            assert "example.test" not in content
            assert "7700" not in content
            assert "555-0101" not in content
            if field == "messages":
                redacted_messages = content
            elif field == "query":
                assert retrieval.query_hash == persistence.keyring.content_hash(
                    content, purpose="query"
                )
            else:
                assert attempt.output_hash == persistence.keyring.content_hash(
                    content, purpose="output"
                )
        assert interaction.input.content_hash == persistence.keyring.content_hash(redacted_messages)
        assert interaction.input.content_hash != persistence.keyring.content_hash(
            json.dumps(body["messages"])
        )
        replay = client.post("/v1/inference", headers=HEADERS, json=body)
        client.app.state.dispatcher.dispatch_once()
        assert replay.json()["replayed"]
        assert "response@example.test" not in replay.json()["content"]
        assert "[REDACTED]" in replay.json()["content"]
        assert app.state.persistence.failures == 0
    database_bytes = b"".join(p.read_bytes() for p in tmp_path.glob("*.sqlite3*"))
    telemetry = caplog.text + " ".join(e.model_dump_json() for e in app.state.events.events)
    for sensitive in (
        prompt,
        result.json()["content"],
        HEADERS["X-Subject"],
        "prompt@example.test",
        "response@example.test",
        "metadata@example.test",
        "+44 7700 900123",
    ):
        assert sensitive.encode() not in database_bytes
        assert sensitive not in telemetry


def test_store_read_delete_isolation_and_tombstones(
    inference_request: InferenceRequest,
) -> None:
    app = create_app(Settings(policy=LoggingPolicy()))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
        )
        client.app.state.dispatcher.dispatch_once()
        iid = response.json()["interaction_id"]
        meta, payloads = app.state.metadata, app.state.payloads
        interaction = meta.get("synthetic-a", Interaction, iid)
        feedback = Feedback(
            interaction_id=iid,
            source="user",
            label_type="thumb",
            value=FeedbackValue(score=1, max_score=1),
        )
        with meta.transaction():
            meta.put("synthetic-a", feedback, now() + timedelta(seconds=60))
        for kind, key in (
            (Interaction, iid),
            (Started, iid),
            (RetrievalRun, interaction.retrieval_run_id),
            (RouteDecision, interaction.route_decision_id),
            (GenerationAttempt, interaction.final_attempt_id),
            (Feedback, feedback.feedback_id),
        ):
            assert meta.get("synthetic-a", kind, key) is not None
            assert meta.get("synthetic-b", kind, key) is None
        ref = interaction.input.messages_ref
        blob = payloads.get("synthetic-a", ref, now())
        assert blob is not None
        assert payloads.get("synthetic-b", ref, now()) is None
        assert meta.state("synthetic-b", iid) is None
        assert not meta.for_subject("synthetic-b", interaction.subject_id_pseudonymous)
        assert (
            meta.get_replay("synthetic-b", "support-assistant", inference_request.request_id, now())
            is None
        )
        path = f"/v1/privacy/interactions/{iid}"
        assert client.delete(path).status_code == 401
        assert client.delete(path, headers=OTHER).status_code == 404
        assert payloads.get("synthetic-a", ref, now()) is not None
        assert client.delete(path, headers=HEADERS).status_code == 204
        assert meta.get_tombstone("synthetic-a", iid) is not None
        assert meta.get_tombstone("synthetic-b", iid) is None
        assert meta.state("synthetic-a", iid) == "deleted"
        assert payloads.get("synthetic-a", ref, now()) is None
        assert meta.get("synthetic-a", Interaction, iid).input.messages_ref is None
        assert meta.get("synthetic-a", RetrievalRun, interaction.retrieval_run_id).query_ref is None
        assert (
            meta.get("synthetic-a", GenerationAttempt, interaction.final_attempt_id).output_ref
            is None
        )
        assert (
            meta.get_replay("synthetic-a", "support-assistant", inference_request.request_id, now())
            is None
        )
        app.state.dispatcher.dispatch_once()
        assert app.state.events.events[-1].event_type == "privacy.deletion.requested.v1"
        assert app.state.events.events[-1].trace_id == interaction.trace_id
        for write in (
            lambda: meta.put("synthetic-a", interaction, now() + timedelta(seconds=60)),
            lambda: payloads.put(blob),
        ):
            with pytest.raises(StorageError, match="^interaction_deleted$"):
                with meta.transaction():
                    write()
        assert client.delete(path, headers=HEADERS).status_code == 204
        retry = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        client.app.state.dispatcher.dispatch_once()
        assert retry.status_code == 200
        assert retry.json()["interaction_id"] != iid


def test_subject_deletion_is_pseudonymous_and_tenant_scoped(
    inference_request: InferenceRequest,
) -> None:
    app = create_app(Settings(policy=LoggingPolicy()))
    with TestClient(app) as client:
        ids = []
        for index in range(2):
            body = {**inference_request.model_dump(), "request_id": f"synthetic-{index}"}
            ids.append(
                client.post("/v1/inference", headers=HEADERS, json=body).json()["interaction_id"]
            )
        other = client.post(
            "/v1/inference", headers=OTHER, json=inference_request.model_dump()
        ).json()
        untouched = client.post(
            "/v1/inference",
            headers={**HEADERS, "X-Subject": "synthetic-another"},
            json={**inference_request.model_dump(), "request_id": "other-subject"},
        )
        client.app.state.dispatcher.dispatch_once()
        result = client.post(
            "/v1/privacy/subjects/deletion-requests",
            headers=HEADERS,
            json={"subject": "synthetic-raw-subject"},
        )
        client.app.state.dispatcher.dispatch_once()
        assert result.status_code == 200
        assert result.json() == {"deleted": 2}
        for iid in ids:
            assert app.state.metadata.state("synthetic-a", iid) == "deleted"
        assert app.state.metadata.state("synthetic-b", other["interaction_id"]) == "active"
        assert (
            app.state.metadata.state("synthetic-a", untouched.json()["interaction_id"]) == "active"
        )
        events = [
            e for e in app.state.events.events if e.event_type == "privacy.deletion.requested.v1"
        ]
        assert len(events) == 3
        assert {e.data.target_id for e in events if e.data.scope == "interaction"} == set(ids)
        subject_event = events[-1]
        assert subject_event.data.scope == "subject"
        pseudonym = app.state.keyring.pseudonym("synthetic-a", "synthetic-raw-subject")
        assert subject_event.data.target_id == pseudonym
        assert (
            app.state.metadata.get_subject_tombstone("synthetic-a", pseudonym) == subject_event.data
        )
        assert app.state.metadata.get_subject_tombstone("synthetic-b", pseudonym) is None
        late = app.state.metadata.get("synthetic-a", Interaction, ids[0]).model_copy(
            update={"interaction_id": uid()}
        )
        with pytest.raises(StorageError, match="^subject_deleted$"):
            with app.state.metadata.transaction():
                app.state.metadata.put("synthetic-a", late, now() + timedelta(seconds=60))
        assert app.state.metadata.get("synthetic-a", Interaction, late.interaction_id) is None
        assert "/v1/privacy/subjects/deletion-requests" in app.openapi()["paths"]
        assert all(
            "{subject" not in path and "synthetic-raw-subject" not in path
            for path in app.openapi()["paths"]
        )
        assert all("synthetic-raw-subject" not in e.model_dump_json() for e in events)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"subject": ""},
        {"subject": 123},
        {"subject": "x" * 513},
        {"subject": "synthetic-raw-subject", "tenant_id": "synthetic-b"},
    ],
)
def test_subject_deletion_body_is_strict_and_bounded(body: dict[str, object]) -> None:
    with TestClient(create_app()) as client:
        result = client.post("/v1/privacy/subjects/deletion-requests", headers=HEADERS, json=body)
    assert result.status_code == 422
    assert result.json() == {"error": {"code": "invalid_request"}}


def test_subject_tombstone_without_existing_interactions_survives_restart(
    tmp_path: Path, inference_request: InferenceRequest, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        result = client.post(
            "/v1/privacy/subjects/deletion-requests",
            headers=HEADERS,
            json={"subject": "synthetic-raw-subject"},
        )
        client.app.state.dispatcher.dispatch_once()
        assert result.json() == {"deleted": 0}
        assert len(app.state.events.events) == 1
        assert app.state.events.events[0].data.scope == "subject"
    restarted = create_app(settings)
    with TestClient(restarted) as client:
        response = client.post(
            "/v1/inference", headers=HEADERS, json=inference_request.model_dump()
        )
        client.app.state.dispatcher.dispatch_once()
        assert response.status_code == 200
        assert (
            restarted.state.metadata.get(
                "synthetic-a", Interaction, response.json()["interaction_id"]
            )
            is None
        )
        assert restarted.state.persistence.failures == 1
        assert (
            restarted.state.metadata.get_replay(
                "synthetic-a", "support-assistant", inference_request.request_id, now()
            )
            is None
        )
    assert "synthetic-raw-subject" not in caplog.text


def test_subject_deletion_rolls_back_tombstones_and_payload_deletes(
    tmp_path: Path, inference_request: InferenceRequest
) -> None:
    from adaptive_llm.storage.sqlite import SQLiteDatabase, SQLiteMetadataStore, SQLitePayloadStore

    database = SQLiteDatabase(tmp_path)
    metadata = SQLiteMetadataStore(database)

    class BrokenDelete(SQLitePayloadStore):
        def delete_interaction(self, tenant_id: str, interaction_id: str) -> None:
            super().delete_interaction(tenant_id, interaction_id)
            raise StorageError("synthetic_delete_failed")

    payloads = BrokenDelete(database)
    app = create_app(
        Settings(metadata_store=metadata, payload_store=payloads, policy=LoggingPolicy())
    )
    with TestClient(app) as client:
        result = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        iid = result.json()["interaction_id"]
        pseudonym = app.state.keyring.pseudonym("synthetic-a", "synthetic-raw-subject")
        with pytest.raises(StorageError, match="^synthetic_delete_failed$"):
            app.state.persistence.delete_subject("synthetic-a", pseudonym, None)
        assert metadata.get_subject_tombstone("synthetic-a", pseudonym) is None
        assert metadata.get_tombstone("synthetic-a", iid) is None
        interaction = metadata.get("synthetic-a", Interaction, iid)
        assert interaction is not None and interaction.input.messages_ref is not None
        assert payloads.get("synthetic-a", interaction.input.messages_ref, now()) is not None
    database.close()


@pytest.mark.parametrize("failing_pass", ["input", "output"])
def test_redaction_failure_still_serves_but_writes_no_payloads_or_replay(
    inference_request: InferenceRequest,
    caplog: pytest.LogCaptureFixture,
    failing_pass: str,
) -> None:
    class BrokenRedactor:
        version = "synthetic-failed-redactor"

        def redact_text(self, content: str, policy: PolicyDecision) -> tuple[str, dict[str, int]]:
            is_output = content.startswith("SYNTHETIC ANSWER:")
            if is_output == (failing_pass == "output"):
                raise RuntimeError("SYNTHETIC-private-redactor-error")
            return content, {}

    app = create_app(Settings(policy=LoggingPolicy(), persistence_redactor=BrokenRedactor()))
    with TestClient(app) as client:
        result = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        client.app.state.dispatcher.dispatch_once()
        assert result.status_code == 200
        assert result.json()["content"]
        meta = app.state.metadata
        interaction = meta.get("synthetic-a", Interaction, result.json()["interaction_id"])
        assert interaction.status == "completed"
        assert interaction.error_code == "persistence_redaction_failed"
        assert interaction.input.messages_ref is None
        if failing_pass == "input":
            assert interaction.input.content_hash is None
            assert (
                meta.get("synthetic-a", RetrievalRun, interaction.retrieval_run_id).query_hash
                is None
            )
        assert interaction.policy.persistence_redaction_version == "synthetic-failed-redactor"
        assert meta.get("synthetic-a", RetrievalRun, interaction.retrieval_run_id).query_ref is None
        assert (
            meta.get("synthetic-a", GenerationAttempt, interaction.final_attempt_id).output_hash
            is None
        )
        assert (
            meta.get_replay("synthetic-a", "support-assistant", inference_request.request_id, now())
            is None
        )
        assert app.state.events.events[1].data == meta.get(
            "synthetic-a", RetrievalRun, interaction.retrieval_run_id
        )
        assert app.state.events.events[3].data == meta.get(
            "synthetic-a", GenerationAttempt, interaction.final_attempt_id
        )
        assert (
            app.state.database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0]
            == 0
        )
        assert app.state.persistence.failures == 0
        assert "SYNTHETIC-private-redactor-error" not in caplog.text
        assert all(
            "SYNTHETIC-private-redactor-error" not in e.model_dump_json()
            for e in app.state.events.events
        )


def test_retention_removes_all_payloads_and_clears_refs(
    inference_request: InferenceRequest,
) -> None:
    app = create_app(Settings(policy=LoggingPolicy()))
    at = now()
    with TestClient(app) as client:
        app.state.persistence.clock = lambda: at
        result = client.post("/v1/inference", headers=HEADERS, json=inference_request.model_dump())
        iid = result.json()["interaction_id"]
        meta, payloads = app.state.metadata, app.state.payloads
        assert (
            app.state.database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0]
            == 4
        )
        assert sweep(meta, payloads, "synthetic-a", clock=lambda: at + timedelta(seconds=60)) == 1
        interaction = meta.get("synthetic-a", Interaction, iid)
        assert interaction.input.messages_ref is None
        assert meta.get("synthetic-a", RetrievalRun, interaction.retrieval_run_id).query_ref is None
        assert (
            meta.get("synthetic-a", GenerationAttempt, interaction.final_attempt_id).output_ref
            is None
        )
        assert (
            app.state.database.connection.execute("SELECT count(*) FROM payloads").fetchone()[0]
            == 0
        )
