import base64
import hashlib
import json
import os
import subprocess

import pytest
from pydantic import ValidationError

from adaptive_llm.app import Settings
from adaptive_llm.gateway.identity import GatewayError, KeyIdentity, LocalAuthenticator


@pytest.mark.parametrize("encoding", ["hex", "base64"])
@pytest.mark.parametrize("file_secret", [None, "SYNTHETIC-LOCAL-ONLY-hmac-secret-v1"])
def test_environment_secret_overrides_file_and_stays_private(
    tmp_path, monkeypatch, encoding, file_secret
):
    secret = b"SYNTHETIC-private-environment-secret-32"
    encoded = secret.hex() if encoding == "hex" else base64.b64encode(secret).decode()
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"local_secret": file_secret, "keys": {}}))
    monkeypatch.setenv("ADAPTIVE_SECRET", encoded)
    settings = Settings(identity_path=path, environment="production", payload_key=b"x" * 32)
    assert settings.secret == secret
    assert encoded not in repr(settings) and secret.decode() not in repr(settings)


@pytest.mark.parametrize(
    "encoded", ["", "not-a-secret!", "ab" * 31, base64.b64encode(b"a" * 31).decode()]
)
def test_invalid_environment_secret_is_fixed(monkeypatch, encoded):
    monkeypatch.setenv("ADAPTIVE_SECRET", encoded)
    with pytest.raises(ValueError, match="^invalid_adaptive_secret$"):
        Settings()


@pytest.mark.parametrize("environment", ["local", "development", "staging", "production"])
@pytest.mark.parametrize("hashed", [False, True])
def test_synthetic_secret_startup_guard(tmp_path, environment, hashed):
    path = tmp_path / "identity.json"
    entry = {"tenant_id": "synthetic", "application_ids": ["app"], "environment": "local"}
    if hashed:
        entry["key_sha256"] = "a" * 64
    path.write_text(
        json.dumps({"local_secret": "SYNTHETIC-LOCAL-ONLY-hmac-secret-v1", "keys": {"key": entry}})
    )
    options = dict(identity_path=path, environment=environment, payload_key=b"x" * 32)
    if environment == "local" and not hashed:
        assert Settings(**options).secret == b"SYNTHETIC-LOCAL-ONLY-hmac-secret-v1"
    else:
        with pytest.raises(ValueError, match="^insecure_adaptive_secret$"):
            Settings(**options)
    assert Settings(**options, secret=b"custom-synthetic-test-secret-32-bytes").secret


def test_manyfails_requires_secret_and_env_accepts_hashed_identity(monkeypatch):
    from adaptive_llm.app import ROOT

    path = ROOT / "configs/identity/manyfails.json"
    assert "local_secret" not in json.loads(path.read_text())
    with pytest.raises(ValueError, match="^adaptive_secret_required$"):
        Settings(identity_path=path)
    monkeypatch.setenv("ADAPTIVE_SECRET", (b"x" * 32).hex())
    assert Settings(identity_path=path).secret == b"x" * 32
    assert Settings(identity_path=path, secret=None).secret == b"x" * 32
    monkeypatch.setenv("ADAPTIVE_SECRET", b"SYNTHETIC-LOCAL-ONLY-hmac-secret-v1".hex())
    with pytest.raises(ValueError, match="^insecure_adaptive_secret$"):
        Settings(identity_path=path)


def test_custom_file_secret_is_retained_for_compatibility(tmp_path):
    path = tmp_path / "identity.json"
    secret = "SYNTHETIC-custom-file-secret-32-bytes"
    path.write_text(json.dumps({"local_secret": secret, "keys": {}}))
    settings = Settings(identity_path=path, environment="production", payload_key=b"x" * 32)
    assert settings.secret == secret.encode() and secret not in repr(settings)


def test_hashed_literal_identity_and_constant_time_comparisons(tmp_path, keyring, monkeypatch):
    key = "SYNTHETIC-private-key"
    config = {
        "local_secret": "synthetic",
        "keys": {
            "hashed-label": {
                "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "tenant_id": "manyfails",
                "application_ids": ["research-sweep"],
                "environment": "local",
            },
            "synthetic-literal": {
                "tenant_id": "literal",
                "application_ids": ["local"],
                "environment": "local",
            },
        },
    }
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(config))
    import adaptive_llm.gateway.identity as module

    original = module.hmac.compare_digest
    seen = []

    def compare(a, b):
        seen.append((len(a), len(b)))
        return original(a, b)

    monkeypatch.setattr(module.hmac, "compare_digest", compare)
    auth = LocalAuthenticator(path, keyring)
    for header, tenant in (
        (f"Bearer {key}", "manyfails"),
        ("Bearer synthetic-literal", "literal"),
        ("Bearer wrong", None),
        ("Bearer hashed-label", None),
        ("Basic " + key, None),
    ):
        seen.clear()
        if tenant:
            assert auth.authenticate(header, None).tenant_id == tenant
        else:
            with pytest.raises(GatewayError, match="^unauthenticated$"):
                auth.authenticate(header, None)
        assert seen == [(64, 64), (64, 64)]
    with pytest.raises(ValidationError):
        KeyIdentity(**{**config["keys"]["hashed-label"], "key_sha256": "bad"})


def test_issue_key_make_prints_once_and_authenticates(tmp_path, keyring):
    result = subprocess.run(
        ["make", "--silent", "issue-key", "TENANT=manyfails", "APP=research-sweep"],
        capture_output=True,
        text=True,
        env={**os.environ, "UV_OFFLINE": "1"},
        check=True,
    )
    key, encoded = result.stdout.split("\n", 1)
    stanza = json.loads(encoded)
    assert len(key) >= 40 and result.stdout.count(key) == 1 and key not in encoded
    entry = stanza["manyfails-research-sweep"]
    assert entry["key_sha256"] == hashlib.sha256(key.encode()).hexdigest()
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({"local_secret": "synthetic", "keys": stanza}))
    identity = LocalAuthenticator(path, keyring).authenticate(f"Bearer {key}", None)
    assert identity.tenant_id == "manyfails" and identity.application_ids == frozenset(
        {"research-sweep"}
    )


def test_key_environment_not_repr(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC-PRIVATE-ENV-KEY")
    settings = Settings()
    assert settings.openrouter_api_key == "SYNTHETIC-PRIVATE-ENV-KEY"
    assert settings.openrouter_api_key not in repr(settings)
