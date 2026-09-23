"""Authenticated encryption with tenant, interaction and field binding."""

import json
import secrets
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from adaptive_llm.contracts import uid
from adaptive_llm.storage import EncryptedPayload, StorageError


def load_keyring(path: Path) -> tuple[dict[str, bytes], str]:
    """Read operator-provisioned keys; no file contents escape in validation errors."""
    try:
        config = json.loads(path.read_text())
        keys = {version: bytes.fromhex(value) for version, value in config["keys"].items()}
        current = config["current_version"]
        if not isinstance(current, str):
            raise ValueError
        PayloadCipher(keys, current)
        return keys, current
    except Exception:
        raise StorageError("invalid_payload_keyring") from None


class PayloadCipher:
    def __init__(self, key: bytes | Mapping[str, bytes], key_version: str = "local-1") -> None:
        keys = {key_version: key} if isinstance(key, bytes) else dict(key)
        if not keys or any(not version or len(value) != 32 for version, value in keys.items()):
            raise ValueError("invalid_payload_key")
        if key_version not in keys:
            raise ValueError("unknown_current_key_version")
        self._ciphers = {version: AESGCM(value) for version, value in keys.items()}
        self.key_version = key_version

    @staticmethod
    def _aad(tenant_id: str, interaction_id: str, field: str) -> bytes:
        if any("|" in part for part in (tenant_id, interaction_id, field)):
            raise StorageError("invalid_payload_binding")
        return f"{tenant_id}|{interaction_id}|{field}".encode()

    def encrypt(
        self,
        content: bytes,
        tenant_id: str,
        interaction_id: str,
        field: str,
        expires_at: datetime,
        *,
        aad_field: str | None = None,
    ) -> EncryptedPayload:
        nonce = secrets.token_bytes(12)
        return EncryptedPayload(
            reference=uid(),
            tenant_id=tenant_id,
            interaction_id=interaction_id,
            field=field,
            nonce=nonce,
            ciphertext=self._ciphers[self.key_version].encrypt(
                nonce, content, self._aad(tenant_id, interaction_id, aad_field or field)
            ),
            key_version=self.key_version,
            expires_at=expires_at,
        )

    def decrypt(
        self,
        payload: EncryptedPayload,
        tenant_id: str,
        interaction_id: str,
        field: str,
        *,
        aad_field: str | None = None,
    ) -> bytes:
        if payload.key_version not in self._ciphers:
            raise StorageError("unknown_payload_key_version")
        if payload.field != field:
            raise StorageError("payload_authentication_failed")
        try:
            return self._ciphers[payload.key_version].decrypt(
                payload.nonce,
                payload.ciphertext,
                self._aad(tenant_id, interaction_id, aad_field or field),
            )
        except InvalidTag:
            raise StorageError("payload_authentication_failed") from None
