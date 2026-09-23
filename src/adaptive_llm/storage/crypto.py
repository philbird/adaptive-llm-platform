"""Authenticated encryption with tenant, interaction and field binding."""

import secrets
from datetime import datetime

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from adaptive_llm.contracts import uid
from adaptive_llm.storage import EncryptedPayload, StorageError


class PayloadCipher:
    def __init__(self, key: bytes, key_version: str = "local-1") -> None:
        if len(key) != 32:
            raise ValueError("invalid_payload_key")
        self._cipher = AESGCM(key)
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
    ) -> EncryptedPayload:
        nonce = secrets.token_bytes(12)
        return EncryptedPayload(
            reference=uid(),
            tenant_id=tenant_id,
            interaction_id=interaction_id,
            field=field,
            nonce=nonce,
            ciphertext=self._cipher.encrypt(
                nonce, content, self._aad(tenant_id, interaction_id, field)
            ),
            key_version=self.key_version,
            expires_at=expires_at,
        )

    def decrypt(
        self, payload: EncryptedPayload, tenant_id: str, interaction_id: str, field: str
    ) -> bytes:
        if payload.key_version != self.key_version:
            raise StorageError("unknown_payload_key_version")
        try:
            return self._cipher.decrypt(
                payload.nonce, payload.ciphertext, self._aad(tenant_id, interaction_id, field)
            )
        except InvalidTag:
            raise StorageError("payload_authentication_failed") from None
