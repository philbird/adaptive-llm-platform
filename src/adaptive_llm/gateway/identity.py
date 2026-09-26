"""Synthetic local API keys; callers cannot choose their tenant or environment."""

import argparse
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from adaptive_llm.contracts import Environment, Identifier, SignedRecord
from adaptive_llm.signing import Signer, Verifier, sign_record, signed, verify_record


class GatewayError(Exception):
    """Only fixed, content-free codes may cross an HTTP boundary."""

    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class Keyring:
    """Derive separate HMAC keys once; never use the master secret to hash caller content."""

    def __init__(
        self,
        secret: bytes,
        *,
        signer: Signer | None = None,
        verifier: Verifier | None = None,
        legacy_mac_records: bool = False,
    ) -> None:
        self.signer, self.verifier = signer, verifier
        self.legacy_mac_records = legacy_mac_records
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
                "evaluation-report-v1",
                "model-artifact-v1",
            )
        }

    def require_signer(self) -> None:
        if self.verifier is not None and self.signer is None:
            raise GatewayError(503, "signing_key_required")

    @property
    def accepts_legacy_mac(self) -> bool:
        return self.verifier is None or self.legacy_mac_records

    def checkpoint_seal(self, payload: str) -> dict[str, str]:
        self.require_signer()
        if self.signer is None:
            return {"mac": self.artifact_mac(payload)}
        record = sign_record(SignedRecord(), self.signer, "training-checkpoint", payload)
        return {
            "signature": record.signature or "",
            "signature_key_id": record.signature_key_id or "",
            "signature_version": record.signature_version or "",
            "mac": "",
        }

    def verify_checkpoint(self, seal: dict[str, str], payload: str) -> None:
        record = SignedRecord.model_validate(seal)
        if signed(record):
            verify_record(record, self.verifier, "training-checkpoint", payload)
        elif not self.accepts_legacy_mac or not hmac.compare_digest(
            seal.get("mac", ""), self.artifact_mac(payload)
        ):
            raise ValueError("checkpoint_integrity_failed")

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

    def report_mac(self, value: str) -> str:
        return self._hash("evaluation-report-v1", value)

    def artifact_mac(self, value: str) -> str:
        return self._hash("model-artifact-v1", value)


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    application_ids: frozenset[str]
    environment: Environment
    subject_id_pseudonymous: str | None
    key_class: Literal["user", "operator"] = "user"
    dataset_tenants: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()


class Authenticator(Protocol):
    def authenticate(self, authorization: str | None, subject: str | None) -> Identity: ...


class KeyIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$", repr=False)
    tenant_id: Identifier
    application_ids: frozenset[Identifier]
    environment: Environment
    key_class: Literal["user", "operator"] = "user"
    dataset_tenants: frozenset[Identifier] = frozenset()
    capabilities: frozenset[Identifier] = frozenset()


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    local_secret: str | None = Field(default=None, repr=False)
    keys: dict[str, KeyIdentity]


class LocalAuthenticator:
    def __init__(self, path: Path, keyring: Keyring) -> None:
        config = IdentityConfig.model_validate_json(path.read_text())
        self._keyring = keyring
        self._keys = [
            (entry.key_sha256 or hashlib.sha256(key.encode()).hexdigest(), entry)
            for key, entry in config.keys.items()
        ]
        if len({digest for digest, _ in self._keys}) != len(self._keys):
            raise ValueError("duplicate_identity_key")

    def authenticate(self, authorization: str | None, subject: str | None) -> Identity:
        scheme, _, key = (authorization or "").partition(" ")
        presented = hashlib.sha256(key.encode()).hexdigest()
        identity = None
        # Always scan every fixed-length digest; neither key position nor a prefix affects lookup.
        for digest, entry in self._keys:
            matched = hmac.compare_digest(presented, digest)
            if matched and scheme.lower() == "bearer":
                identity = entry
        if identity is None:
            raise GatewayError(401, "unauthenticated")
        pseudonym = (
            self._keyring.pseudonym(identity.tenant_id, subject if subject is not None else key)
            if subject is not None or identity.key_class == "operator"
            else None
        )
        return Identity(
            tenant_id=identity.tenant_id,
            application_ids=identity.application_ids,
            environment=identity.environment,
            subject_id_pseudonymous=pseudonym,
            key_class=identity.key_class,
            dataset_tenants=identity.dataset_tenants,
            capabilities=identity.capabilities,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Issue a bearer key and its hashed identity stanza"
    )
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--app", required=True)
    args = parser.parse_args(argv)
    key = secrets.token_urlsafe(32)
    try:
        entry = KeyIdentity(
            tenant_id=args.tenant,
            application_ids=frozenset({args.app}),
            environment="local",
            key_sha256=hashlib.sha256(key.encode()).hexdigest(),
        )
    except Exception:
        parser.exit(1, "invalid_key_identity\n")
    print(key)
    print(json.dumps({f"{entry.tenant_id}-{args.app}": entry.model_dump(mode="json")}, indent=2))


if __name__ == "__main__":
    main()
