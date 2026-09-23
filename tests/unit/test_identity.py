import hashlib
import hmac
import json

import pytest

from adaptive_llm.app import Settings
from adaptive_llm.gateway.identity import GatewayError, Keyring, LocalAuthenticator


def test_local_identity_is_tenant_scoped_hmac(settings: Settings, keyring: Keyring) -> None:
    auth = LocalAuthenticator(settings.identity_path, keyring)
    a = auth.authenticate("Bearer synthetic-key-a", "synthetic-subject")
    b = auth.authenticate("Bearer synthetic-key-b", "synthetic-subject")
    assert a.tenant_id == "synthetic-a"
    assert a.application_ids == frozenset({"support-assistant"})
    assert a.environment == "local"
    assert settings.secret is not None
    derived_key = hmac.new(settings.secret, b"subject-pseudonym-v1", hashlib.sha256).digest()
    assert (
        a.subject_id_pseudonymous
        == hmac.new(
            derived_key,
            json.dumps(["synthetic-a", "synthetic-subject"]).encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    assert a.subject_id_pseudonymous != b.subject_id_pseudonymous
    assert auth.authenticate("bearer synthetic-key-a", None).subject_id_pseudonymous is None


@pytest.mark.parametrize("header", [None, "", "Basic synthetic-key-a", "Bearer unknown"])
def test_invalid_keys_fail_closed(settings: Settings, keyring: Keyring, header: str | None) -> None:
    with pytest.raises(GatewayError, match="^unauthenticated$") as failure:
        LocalAuthenticator(settings.identity_path, keyring).authenticate(header, None)
    assert failure.value.status_code == 401


def test_keyring_separates_all_hash_purposes() -> None:
    secret = b"synthetic-master-secret"
    keyring = Keyring(secret)
    value = json.dumps(["synthetic-a", "synthetic-subject"])
    hashes = {
        "subject-pseudonym-v1": keyring.pseudonym("synthetic-a", "synthetic-subject"),
        "input-content-v1": keyring.content_hash(value, purpose="input"),
        "output-content-v1": keyring.content_hash(value, purpose="output"),
        "query-content-v1": keyring.content_hash(value, purpose="query"),
        "replay-fingerprint-v1": keyring.fingerprint(value),
    }
    assert len(set(hashes.values())) == 5
    assert hmac.new(secret, value.encode(), hashlib.sha256).hexdigest() not in hashes.values()
    for purpose, digest in hashes.items():
        derived_key = hmac.new(secret, purpose.encode(), hashlib.sha256).digest()
        assert digest == hmac.new(derived_key, value.encode(), hashlib.sha256).hexdigest()
    assert keyring.fingerprint(value) == Keyring(secret).fingerprint(value)
    assert keyring.fingerprint(value) != Keyring(b"synthetic-other-secret").fingerprint(value)
