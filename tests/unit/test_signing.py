import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from adaptive_llm.app import Settings
from adaptive_llm.contracts import SignedRecord
from adaptive_llm.gateway.identity import Keyring
from adaptive_llm.signing import (
    Ed25519Signer,
    Ed25519Verifier,
    local_keys,
    rotate,
    sign_record,
    verify_record,
)
from adaptive_llm.signing.__main__ import main


def test_rotation_preserves_public_verification_and_cli(tmp_path, monkeypatch, capsys):
    public = tmp_path / "public.json"
    a, b = tmp_path / "a.pem", tmp_path / "b.pem"
    rotate(a, public, "a")
    first = Ed25519Signer(a, "a")
    record = sign_record(SignedRecord(), first, "synthetic-purpose", "synthetic-payload")
    monkeypatch.setattr(
        sys,
        "argv",
        ["sign-rotate", "--private-key", str(b), "--public-keys", str(public), "--key-id", "b"],
    )
    main()
    assert capsys.readouterr().out == "signing_key_rotated\n"
    ring = json.loads(public.read_text())
    assert ring["active_key_id"] == "b"
    assert ring["keys"]["a"]["status"] == "verify-only"
    second = Ed25519Signer(b, "b")
    newer = sign_record(SignedRecord(), second, "synthetic-purpose", "synthetic-payload")
    assert newer.signature_key_id == "b"
    a.unlink()
    b.unlink()
    verifier = Ed25519Verifier.load(public)
    for value in (record, newer):
        verify_record(value, verifier, "synthetic-purpose", "synthetic-payload")
    for change in (
        {"signature_key_id": "unknown"},
        {"signature": "bad"},
        {"signature_version": None},
    ):
        with pytest.raises(ValueError, match="signature_verification_failed"):
            verify_record(
                record.model_copy(update=change), verifier, "synthetic-purpose", "synthetic-payload"
            )
    with pytest.raises(ValueError):
        verify_record(record, verifier, "another-purpose", "synthetic-payload")
    with pytest.raises(ValueError):
        verify_record(record, verifier, "synthetic-purpose", "changed")
    with pytest.raises(ValueError, match="signing_key_exists"):
        rotate(a, public, "a")


def test_settings_load_pem_and_verifier_only_checkpoint(tmp_path: Path):
    private, public = tmp_path / "key.pem", tmp_path / "keys.json"
    rotate(private, public, "synthetic")
    settings = Settings(
        signing_key_path=private, signing_key_id="synthetic", signing_public_keys_path=public
    )
    writer = Keyring(b"synthetic", signer=settings.signer, verifier=settings.verifier)
    seal = writer.checkpoint_seal("synthetic-hash")
    private.unlink()
    verifier_only = Settings(signing_public_keys_path=public)
    reader = Keyring(b"different-synthetic-secret", verifier=verifier_only.verifier)
    assert reader.signer is None
    reader.verify_checkpoint(seal, "synthetic-hash")
    with pytest.raises(Exception, match="signing_key_required"):
        reader.checkpoint_seal("synthetic-hash")
    with pytest.raises(ValueError):
        reader.verify_checkpoint(seal, "changed")
    legacy = Keyring(b"synthetic")
    legacy_seal = legacy.checkpoint_seal("legacy")
    legacy.verify_checkpoint(legacy_seal, "legacy")
    with pytest.raises(ValueError, match="signature_verification_failed"):
        legacy.verify_checkpoint({**legacy_seal, "signature": ""}, "legacy")
    with pytest.raises(ValueError, match="invalid_signing_key"):
        Ed25519Signer(private, "missing")


def test_local_key_restart_and_no_fixed_test_key(tmp_path):
    a, av = local_keys(tmp_path / "a")
    again, _ = local_keys(tmp_path / "a")
    b, _ = local_keys(tmp_path / "b")
    assert a.sign(b"synthetic") == again.sign(b"synthetic")
    assert a.sign(b"synthetic") != b.sign(b"synthetic")
    av.verify(b"synthetic", a.sign(b"synthetic"), a.key_id, "ed25519-v1")
    key = serialization.load_pem_private_key((tmp_path / "a/private.pem").read_bytes(), None)
    assert isinstance(key, Ed25519PrivateKey)


def test_retired_or_mismatched_signing_key_is_rejected(tmp_path):
    private, next_key, public = tmp_path / "a.pem", tmp_path / "b.pem", tmp_path / "public.json"
    rotate(private, public, "a")
    rotate(next_key, public, "b")
    with pytest.raises(ValueError, match="inactive_signing_key"):
        Settings(signing_key_path=private, signing_key_id="a", signing_public_keys_path=public)
    with pytest.raises(ValueError, match="signature_verification_failed"):
        Settings(signing_key_path=private, signing_key_id="b", signing_public_keys_path=public)


@pytest.mark.parametrize("environment", ["local", "development", "staging", "production"])
def test_legacy_mac_defaults_and_environment_override(environment, monkeypatch):
    monkeypatch.delenv("LEGACY_MAC_RECORDS", raising=False)
    options = dict(environment=environment, payload_key=b"x" * 32, secret=b"s" * 32)
    assert Settings(**options).legacy_mac_records is (environment == "local")
    for value, expected in (("true", True), ("1", True), ("false", False), ("0", False)):
        monkeypatch.setenv("LEGACY_MAC_RECORDS", value)
        assert Settings(**options).legacy_mac_records is expected
        assert (
            Settings(**options, legacy_mac_records=not expected).legacy_mac_records is not expected
        )
    monkeypatch.setenv("LEGACY_MAC_RECORDS", "invalid")
    with pytest.raises(ValueError, match="invalid_legacy_mac_records"):
        Settings(**options)
