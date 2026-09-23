"""Synthetic local API keys; callers cannot choose their tenant or environment."""

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from adaptive_llm.contracts import Environment, Identifier


class GatewayError(Exception):
    """Only fixed, content-free codes may cross an HTTP boundary."""

    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class Keyring:
    """Derive separate HMAC keys once; never use the master secret to hash caller content."""

    def __init__(self, secret: bytes) -> None:
        self._keys = {
            purpose: hmac.new(secret, purpose.encode("utf-8"), hashlib.sha256).digest()
            for purpose in (
                "subject-pseudonym-v1",
                "input-content-v1",
                "output-content-v1",
                "query-content-v1",
                "replay-fingerprint-v1",
                "dataset-example-v1",
                "dataset-manifest-v1",
            )
        }

    def _hash(self, purpose: str, value: str) -> str:
        return hmac.new(self._keys[purpose], value.encode("utf-8"), hashlib.sha256).hexdigest()

    def pseudonym(self, tenant_id: str, subject: str) -> str:
        return self._hash("subject-pseudonym-v1", json.dumps([tenant_id, subject]))

    def content_hash(
        self, value: str, *, purpose: Literal["input", "output", "query"] = "input"
    ) -> str:
        return self._hash(f"{purpose}-content-v1", value)

    def fingerprint(self, value: str) -> str:
        return self._hash("replay-fingerprint-v1", value)

    def dataset_hash(self, value: str) -> str:
        return self._hash("dataset-example-v1", value)

    def manifest_mac(self, value: str) -> str:
        return self._hash("dataset-manifest-v1", value)


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    application_ids: frozenset[str]
    environment: Environment
    subject_id_pseudonymous: str | None
    key_class: Literal["user", "operator"] = "user"
    dataset_tenants: frozenset[str] = frozenset()


class Authenticator(Protocol):
    def authenticate(self, authorization: str | None, subject: str | None) -> Identity: ...


class KeyIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tenant_id: Identifier
    application_ids: frozenset[Identifier]
    environment: Environment
    key_class: Literal["user", "operator"] = "user"
    dataset_tenants: frozenset[Identifier] = frozenset()


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    local_secret: str
    keys: dict[str, KeyIdentity]


class LocalAuthenticator:
    def __init__(self, path: Path, keyring: Keyring) -> None:
        config = IdentityConfig.model_validate_json(path.read_text())
        self._keyring = keyring
        self._keys = config.keys

    def authenticate(self, authorization: str | None, subject: str | None) -> Identity:
        scheme, _, key = (authorization or "").partition(" ")
        identity = self._keys.get(key) if scheme.lower() == "bearer" else None
        if identity is None:
            raise GatewayError(401, "unauthenticated")
        pseudonym = (
            self._keyring.pseudonym(identity.tenant_id, subject) if subject is not None else None
        )
        return Identity(
            tenant_id=identity.tenant_id,
            application_ids=identity.application_ids,
            environment=identity.environment,
            subject_id_pseudonymous=pseudonym,
            key_class=identity.key_class,
            dataset_tenants=identity.dataset_tenants,
        )
