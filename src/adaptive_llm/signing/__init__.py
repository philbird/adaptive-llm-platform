"""Purpose-bound Ed25519 signatures; verification requires public keys only."""

import base64
import json
import os
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from adaptive_llm.contracts import SignedRecord, uid

VERSION = "ed25519-v1"
FIELDS = {"signature", "signature_key_id", "signature_version"}


class Signer(Protocol):
    @property
    def key_id(self) -> str: ...
    def sign(self, payload: bytes) -> str: ...


class Verifier(Protocol):
    def verify(self, payload: bytes, signature: str, key_id: str, version: str) -> None: ...


class Ed25519Signer:
    def __init__(self, path: Path, key_id: str) -> None:
        try:
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError
        except Exception:
            raise ValueError("invalid_signing_key") from None
        self._key, self.key_id = key, key_id

    def sign(self, payload: bytes) -> str:
        return base64.b64encode(self._key.sign(payload)).decode("ascii")

    def verifier(self) -> "Ed25519Verifier":
        return Ed25519Verifier({self.key_id: self._key.public_key()})


class Ed25519Verifier:
    def __init__(self, keys: dict[str, Ed25519PublicKey]) -> None:
        self._keys = dict(keys)

    @classmethod
    def load(cls, path: Path) -> "Ed25519Verifier":
        try:
            ring = json.loads(path.read_text())
            keys = {}
            for key_id, entry in ring["keys"].items():
                key = serialization.load_pem_public_key(entry["public_key"].encode())
                if not isinstance(key, Ed25519PublicKey):
                    raise ValueError
                keys[key_id] = key
            return cls(keys)
        except Exception:
            raise ValueError("invalid_verification_keys") from None

    def verify(self, payload: bytes, signature: str, key_id: str, version: str) -> None:
        try:
            if version != VERSION:
                raise ValueError
            self._keys[key_id].verify(base64.b64decode(signature, validate=True), payload)
        except Exception:
            raise ValueError("signature_verification_failed") from None


def message(purpose: str, payload: str, key_id: str) -> bytes:
    return json.dumps([VERSION, purpose, key_id, payload], separators=(",", ":")).encode()


def sign_record[T: SignedRecord](record: T, signer: Signer, purpose: str, payload: str) -> T:
    return record.model_copy(
        update={
            "signature": signer.sign(message(purpose, payload, signer.key_id)),
            "signature_key_id": signer.key_id,
            "signature_version": VERSION,
        }
    )


def verify_record(
    record: SignedRecord, verifier: Verifier | None, purpose: str, payload: str
) -> None:
    if verifier is None or not record.signature or not record.signature_key_id:
        raise ValueError("signature_verification_failed")
    verifier.verify(
        message(purpose, payload, record.signature_key_id),
        record.signature,
        record.signature_key_id,
        record.signature_version or "",
    )


def signed(record: SignedRecord) -> bool:
    # Partial signatures must fail closed; never fall back to a legacy MAC.
    return any(
        value is not None
        for value in (record.signature, record.signature_key_id, record.signature_version)
    )


def rotate(private_path: Path, public_path: Path, key_id: str) -> None:
    """Publish the public ring last. Never overwrite an old private key or key id."""
    SignedRecord(signature_key_id=key_id)
    ring = json.loads(public_path.read_text()) if public_path.exists() else {"keys": {}}
    if key_id in ring["keys"] or private_path.exists():
        raise ValueError("signing_key_exists")
    key = Ed25519PrivateKey.generate()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with private_path.open("xb") as stream:
        os.chmod(private_path, 0o600)
        stream.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    for entry in ring["keys"].values():
        entry["status"] = "verify-only"
    ring["keys"][key_id] = {
        "status": "active",
        "public_key": key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode(),
    }
    ring["active_key_id"] = key_id
    public_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = public_path.with_name(f".{uid()}.json")
    temporary.write_text(json.dumps(ring, indent=2))
    temporary.replace(public_path)


def local_keys(directory: Path) -> tuple[Ed25519Signer, Ed25519Verifier]:
    """Development-only persistent keys, generated on first startup outside tracked files."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    private, public = directory / "private.pem", directory / "public.json"
    with (directory / "provision.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not public.exists():
            rotate(private, public, "local-1")
        ring = json.loads(public.read_text())
        signer, verifier = (
            Ed25519Signer(private, ring["active_key_id"]),
            Ed25519Verifier.load(public),
        )
        verifier.verify(
            b"signing-key-check", signer.sign(b"signing-key-check"), signer.key_id, VERSION
        )
        return signer, verifier
