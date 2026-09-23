import json
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from adaptive_llm.contracts import EvaluationReport, Interaction, now, uid
from adaptive_llm.evaluation.runner import isolated_persistence
from adaptive_llm.gateway.identity import GatewayError, Identity

if TYPE_CHECKING:
    from conftest import EvaluationSeed

OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}
USER = {"Authorization": "Bearer synthetic-key-a"}


def test_operator_tenant_scope_and_content_free_report(evaluation_seed: "EvaluationSeed") -> None:
    seed = evaluation_seed
    body = seed.request.model_dump(mode="json")
    assert seed.client.post("/v1/evaluations", json=body).status_code == 401
    assert seed.client.post("/v1/evaluations", json=body, headers=USER).status_code == 403
    limited = Identity(
        "synthetic-a",
        frozenset({"evaluation"}),
        "local",
        None,
        "operator",
        frozenset({"synthetic-a"}),
    )
    with pytest.raises(GatewayError) as error:
        seed.app.state.evaluations.evaluate(seed.request, limited)
    assert error.value.status_code == 404
    body["replace"] = True
    body["operator_note"] = "SYNTHETIC_PRIVATE_OPERATOR_NOTE"
    response = seed.client.post("/v1/evaluations", headers=OPERATOR, json=body)
    assert response.status_code == 200
    report = EvaluationReport.model_validate(response.json())
    encoded = response.content
    for forbidden in [
        b"SYNTHETIC ANSWER",
        b"SYNTHETIC safe",
        b"SYNTHETIC unique",
        b"SYNTHETIC_PRIVATE",
        b"Example Shop",
        b"Ignore evidence",
        b"protected canary",
    ]:
        assert forbidden not in encoded
    db = seed.app.state.evaluation_database.connection
    stored = db.execute("SELECT report FROM evaluation_reports").fetchone()[0].encode()
    assert b"SYNTHETIC_PRIVATE_OPERATOR_NOTE" not in stored
    path = seed.directory / "evaluations" / report.specification.evaluation_id / "report.json"
    assert path.read_bytes() == stored
    assert (
        seed.client.get(
            f"/v1/evaluations/{report.specification.evaluation_id}", headers=USER
        ).status_code
        == 403
    )
    with pytest.raises(GatewayError) as error:
        seed.app.state.evaluations.get(report.specification.evaluation_id, limited)
    assert error.value.status_code == 404


def test_evaluation_policy_no_logging_and_dataset_count_unchanged(
    evaluation_seed: "EvaluationSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = evaluation_seed
    db = seed.app.state.database.connection
    tables = (
        "interactions",
        "started",
        "retrieval_runs",
        "route_decisions",
        "attempts",
        "payloads",
        "replay_entries",
        "outbox",
    )
    before = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in tables}
    attempts_before = seed.app.state.metrics.get("fallback_free_attempts")
    scratch_connections = []

    @contextmanager
    def inspect_scratch(source):
        with isolated_persistence(source) as scratch:
            connection = scratch.metadata.database.connection
            scratch_connections.append(connection)
            assert connection.execute("PRAGMA database_list").fetchone()[2] == ""
            assert scratch.cipher is source.cipher and scratch.keyring is source.keyring
            assert scratch.metadata is not source.metadata
            yield scratch
            evaluations = [
                Interaction.model_validate_json(row[0])
                for row in connection.execute("SELECT data FROM interactions")
            ]
            assert evaluations
            assert all(
                i.application_id == "evaluation"
                and i.policy.policy_version == "evaluation-policy-1"
                and not i.policy.training_allowed
                and not i.policy.content_logging_allowed
                and i.input.messages_ref is None
                for i in evaluations
            )
            assert (
                connection.execute(
                    "SELECT count(*) FROM payloads WHERE field != 'replay'"
                ).fetchone()[0]
                == 0
            )
            assert connection.execute("SELECT count(*) FROM outbox").fetchone()[0] == 0

    monkeypatch.setattr("adaptive_llm.evaluation.service.isolated_persistence", inspect_scratch)
    result = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=seed.request.model_dump(mode="json")
    )
    assert result.status_code == 200
    after = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in tables}
    assert after == before
    assert seed.app.state.metrics.get("fallback_free_attempts") == attempts_before
    assert len(scratch_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        scratch_connections[0].execute("SELECT 1")
    assert not (seed.directory / "evaluations" / seed.request.evaluation_id / "scratch").exists()
    spec = seed.manifest.specification.model_copy(
        update={
            "source_window": seed.manifest.source_window.model_copy(
                update={"end": now() + timedelta(seconds=1)}
            )
        }
    )
    result = seed.client.post(
        "/v1/datasets/builds", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert result.status_code == 200
    assert result.json()["examples"] == seed.manifest.examples
    assert result.json()["content_digest"] == seed.manifest.content_digest
    assert result.json()["quality_summary"]["exclusions"].get("evaluation_interaction", 0) == 0

    # Defence in depth: an old operational row marked as evaluation must stay ineligible.
    row = db.execute("SELECT record_id, data FROM interactions LIMIT 1").fetchone()
    interaction = Interaction.model_validate_json(row["data"])
    db.execute(
        "UPDATE interactions SET data=? WHERE record_id=?",
        (
            interaction.model_copy(update={"application_id": "evaluation"}).model_dump_json(),
            row["record_id"],
        ),
    )
    from adaptive_llm.datasets.eligibility import select

    selection = select(seed.app.state.metadata, seed.app.state.datasets.policy, spec, now())
    assert selection.exclusions["evaluation_interaction"] == 1
    assert all(
        candidate.interaction.interaction_id != interaction.interaction_id
        for candidate in selection.examples
    )


def test_scratch_is_closed_after_suite_failure(
    evaluation_seed: "EvaluationSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = evaluation_seed
    connections = []
    db = seed.app.state.database.connection
    tables = ("interactions", "payloads", "replay_entries", "outbox")
    before = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in tables}

    @contextmanager
    def capture_scratch(source):
        with isolated_persistence(source) as scratch:
            connections.append(scratch.metadata.database.connection)
            yield scratch

    class FailingSuite:
        async def run(self, runner, cases, specification):
            await runner.run(cases[0])
            raise RuntimeError("synthetic_suite_failure")

    monkeypatch.setattr("adaptive_llm.evaluation.service.isolated_persistence", capture_scratch)
    seed.app.state.evaluations.suites["golden"] = FailingSuite()
    response = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=seed.request.model_dump(mode="json")
    )
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "evaluation_failed"}}
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    assert before == {
        table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in tables
    }
    assert not (seed.directory / "evaluations" / seed.request.evaluation_id).exists()


@pytest.mark.parametrize("artifact", ["ciphertext", "hash", "tenant", "manifest", "report"])
def test_tampered_artifacts_fail_closed(evaluation_seed: "EvaluationSeed", artifact: str) -> None:
    seed = evaluation_seed
    directory = seed.directory / "datasets" / seed.manifest.dataset_id / seed.manifest.version
    if artifact == "report":
        result = seed.client.post(
            "/v1/evaluations", headers=OPERATOR, json=seed.request.model_dump(mode="json")
        )
        assert result.status_code == 200
        path = seed.directory / "evaluations" / seed.request.evaluation_id / "report.json"
        path.write_text(path.read_text() + " ")
        response = seed.client.get(
            f"/v1/evaluations/{seed.request.evaluation_id}", headers=OPERATOR
        )
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "evaluation_integrity_failed"}}
        return
    if artifact == "manifest":
        path = directory / "manifest.json"
        path.write_text(path.read_text() + " ")
    else:
        path = directory / "synthetic-a.test.jsonl.enc"
        envelope = json.loads(path.read_text())
        field, value = {
            "ciphertext": ("ciphertext", "AAAA"),
            "hash": ("plaintext_hash", "0" * 64),
            "tenant": ("tenant_id", "synthetic-b"),
        }[artifact]
        envelope[field] = value
        path.write_text(json.dumps(envelope))
    result = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=seed.request.model_dump(mode="json")
    )
    assert result.status_code == 409
    assert result.json() == {"error": {"code": "invalid_dataset_artifact"}}
    assert not list(seed.directory.glob("evaluations/*/report.json"))


def test_validation_errors_do_not_echo_note_or_path(evaluation_seed: "EvaluationSeed") -> None:
    seed = evaluation_seed
    for update in [
        {"evaluation_id": "../SYNTHETIC_PRIVATE_PATH"},
        {"replace": True, "operator_note": ""},
        {"unexpected": "SYNTHETIC_PRIVATE_FIELD"},
    ]:
        response = seed.client.post(
            "/v1/evaluations", headers=OPERATOR, json=seed.request.model_dump(mode="json") | update
        )
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_request"}}
    response = seed.client.get(f"/v1/evaluations/{uid()}", headers=OPERATOR)
    assert response.status_code == 404
