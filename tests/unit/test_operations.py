import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import InferenceRequest, now
from adaptive_llm.storage import EncryptedPayload, StorageError
from adaptive_llm.storage.__main__ import main
from adaptive_llm.storage.crypto import PayloadCipher, load_keyring
from adaptive_llm.storage.operations import backup, restore, rotate_key
from adaptive_llm.storage.sqlite import SQLiteDatabase


def test_keyring_validation_old_decryption_and_missing_version(tmp_path: Path) -> None:
    old = PayloadCipher(b"a" * 32)
    blob = old.encrypt(b"SYNTHETIC payload", "synthetic-a", "synthetic-iid", "output", now())
    rotated = PayloadCipher({"local-1": b"a" * 32, "local-2": b"b" * 32}, "local-2")
    assert (
        rotated.decrypt(blob, blob.tenant_id, blob.interaction_id, blob.field)
        == b"SYNTHETIC payload"
    )
    with pytest.raises(StorageError, match="^unknown_payload_key_version$"):
        PayloadCipher({"local-2": b"b" * 32}, "local-2").decrypt(
            blob, blob.tenant_id, blob.interaction_id, blob.field
        )
    with pytest.raises(ValueError, match="unknown_current_key_version"):
        PayloadCipher({"old": b"a" * 32})
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps({"keys": {"local-1": (b"a" * 32).hex()}, "current_version": "local-1"})
    )
    assert load_keyring(path) == ({"local-1": b"a" * 32}, "local-1")
    assert Settings(payload_keyring_path=path).payload_keys == {"local-1": b"a" * 32}
    path.write_text("SYNTHETIC private malformed keyring")
    with pytest.raises(StorageError, match="^invalid_payload_keyring$"):
        load_keyring(path)


def test_rotation_is_tenant_scoped_preserves_refs_and_rolls_back_failed_batch(
    tmp_path: Path, inference_request: InferenceRequest, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(data_dir=tmp_path, payload_key=b"a" * 32, outbox_dispatch_enabled=False)
    app = create_app(settings)
    with TestClient(app) as client:
        for key in ("a", "b"):
            assert (
                client.post(
                    "/v1/inference",
                    headers={"Authorization": f"Bearer synthetic-key-{key}"},
                    json=inference_request.model_dump(),
                ).status_code
                == 200
            )
        db = app.state.database
        old_rows = db.connection.execute("SELECT * FROM payloads ORDER BY tenant_id").fetchall()
        cipher = PayloadCipher({"local-1": b"a" * 32, "local-2": b"b" * 32}, "local-2")

        class FailsEncrypt(PayloadCipher):
            def encrypt(
                self,
                content: bytes,
                tenant_id: str,
                interaction_id: str,
                field: str,
                expires_at: datetime,
            ) -> EncryptedPayload:
                raise StorageError("synthetic_rotation_failed")

        with pytest.raises(StorageError, match="synthetic_rotation_failed"):
            rotate_key(
                db,
                "synthetic-a",
                FailsEncrypt({"local-1": b"a" * 32, "local-2": b"b" * 32}, "local-2"),
            )
        assert (
            db.connection.execute("SELECT * FROM payloads ORDER BY tenant_id").fetchall()
            == old_rows
        )
        assert rotate_key(db, "synthetic-a", cipher, batch_size=1) == 1
        assert rotate_key(db, "synthetic-a", cipher) == 0
        rows = db.connection.execute("SELECT * FROM payloads ORDER BY tenant_id").fetchall()
        assert rows[0]["reference"] == old_rows[0]["reference"]
        assert rows[0]["nonce"] != old_rows[0]["nonce"]
        assert rows[0]["key_version"] == "local-2"
        assert rows[1] == old_rows[1]
        app.state.persistence.cipher = cipher
        assert client.post(
            "/v1/inference",
            headers={"Authorization": "Bearer synthetic-key-a"},
            json=inference_request.model_dump(),
        ).json()["replayed"]
        keys = tmp_path / "keys.json"
        keys.write_text(
            json.dumps(
                {
                    "keys": {"local-1": (b"a" * 32).hex(), "local-2": (b"b" * 32).hex()},
                    "current_version": "local-2",
                }
            )
        )
        main(
            [
                "rotate-key",
                "--data-dir",
                str(tmp_path),
                "--tenant",
                "synthetic-b",
                "--keyring",
                str(keys),
                "--new-key-version",
                "local-2",
                "--batch-size",
                "1",
            ]
        )
        assert capsys.readouterr().out == "rotated_payloads=1\n"
        with pytest.raises(ValueError, match="invalid_rotation_batch_size"):
            rotate_key(db, "synthetic-a", cipher, batch_size=0)
    restarted = create_app(
        replace(settings, payload_keys={"local-2": b"b" * 32}, payload_key_version="local-2")
    )
    with TestClient(restarted) as client:
        assert client.post(
            "/v1/inference",
            headers={"Authorization": "Bearer synthetic-key-a"},
            json=inference_request.model_dump(),
        ).json()["replayed"]


def test_backup_restore_guards_preserve_existing_database(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "source")
    output = tmp_path / "backup.sqlite3"
    try:
        backup(database, output)
        with pytest.raises(StorageError, match="backup_destination_exists"):
            backup(database, output)
        with pytest.raises(StorageError, match="restore_destination_exists"):
            restore(output, database.path)
        with pytest.raises(StorageError, match="invalid_restore_source"):
            restore(output, output, force=True)
        with pytest.raises(StorageError, match="backup_not_found"):
            restore(tmp_path / "absent", tmp_path / "new")
        before = database.path.read_bytes()
        invalid = tmp_path / "invalid.sqlite3"
        invalid.write_bytes(b"SYNTHETIC malformed backup")
        with pytest.raises(StorageError, match="restore_failed"):
            restore(invalid, database.path, force=True)
        assert database.path.read_bytes() == before
        assert not list(tmp_path.rglob(".restore-*"))
    finally:
        database.close()
