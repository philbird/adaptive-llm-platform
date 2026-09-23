"""Verified encrypted shards and store-only, authenticated dataset approval."""

import base64
import hashlib
import hmac
import json
from pathlib import Path

from adaptive_llm.contracts import DatasetApproval, DatasetManifest, Split, now, uid
from adaptive_llm.datasets.builder import DatasetBuilder, LocalDatasetBuilder
from adaptive_llm.datasets.curation import SPLITS
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.storage import EncryptedPayload
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.storage.persistence import Persistence


def approval_mac(manifest: DatasetManifest, keyring: Keyring) -> str:
    unsigned = manifest.model_copy(
        update={"approval": manifest.approval.model_copy(update={"mac": None})}
    )
    return keyring.manifest_mac(unsigned.model_dump_json())


def read_shards(
    manifest: DatasetManifest, data_dir: Path, cipher: PayloadCipher, keyring: Keyring
) -> dict[Split, list[bytes]]:
    try:
        if any(
            v in {".", ".."} for v in [manifest.dataset_id, manifest.version, *manifest.tenant_ids]
        ):
            raise ValueError
        directory = data_dir / "datasets" / manifest.dataset_id / manifest.version
        encoded = (directory / "manifest.json").read_text()
        original = DatasetManifest.model_validate_json(encoded)
        if (
            not hmac.compare_digest(
                (directory / "manifest.mac").read_text(), keyring.manifest_mac(encoded)
            )
            or original.model_dump(exclude={"approval"})
            != manifest.model_dump(exclude={"approval"})
            or original.approval.status != "pending"
        ):
            raise ValueError
        if manifest.approval.status != "pending" and not hmac.compare_digest(
            manifest.approval.mac or "", approval_mac(manifest, keyring)
        ):
            raise ValueError
        rows: dict[Split, list[bytes]] = {split: [] for split in SPLITS}
        hashes = []
        for tenant in sorted(manifest.tenant_ids):
            for split in SPLITS:
                envelope = json.loads((directory / f"{tenant}.{split}.jsonl.enc").read_text())
                if (envelope["tenant_id"], envelope["split"], envelope["field"]) != (
                    tenant,
                    split,
                    "dataset",
                ):
                    raise ValueError
                binding = f"{manifest.dataset_id}/{manifest.version}"
                blob = EncryptedPayload(
                    reference=uid(),
                    tenant_id=tenant,
                    interaction_id=binding,
                    field="dataset",
                    nonce=base64.b64decode(envelope["nonce"], validate=True),
                    ciphertext=base64.b64decode(envelope["ciphertext"], validate=True),
                    key_version=envelope["key_version"],
                    expires_at=now(),
                )
                plaintext = cipher.decrypt(blob, tenant, binding, "dataset", aad_field=split)
                digest = hashlib.sha256(plaintext).hexdigest()
                if digest != envelope["plaintext_hash"]:
                    raise ValueError
                hashes.append(digest)
                for line in plaintext.splitlines():
                    row = json.loads(line)
                    if row["tenant_id"] != tenant or row["split"] != split:
                        raise ValueError
                    rows[split].append(line)
        if {s: len(r) for s, r in rows.items()} != manifest.examples or hashlib.sha256(
            "".join(hashes).encode()
        ).hexdigest() != manifest.content_digest:
            raise ValueError
        return rows
    except Exception:
        raise GatewayError(409, "invalid_dataset_artifact") from None


class DatasetApprovals:
    def __init__(self, builder: DatasetBuilder, persistence: Persistence, data_dir: Path) -> None:
        self.builder, self.persistence, self.data_dir = builder, persistence, data_dir

    def approve(
        self, dataset_id: str, version: str, identity: Identity, reason: str
    ) -> DatasetManifest:
        LocalDatasetBuilder._authorize(identity, [])
        if not identity.subject_id_pseudonymous or not reason.strip():
            raise GatewayError(422, "operator_note_required")
        manifest = self.builder.get(dataset_id, version, identity)
        read_shards(manifest, self.data_dir, self.persistence.cipher, self.persistence.keyring)
        with self.persistence.metadata.transaction():
            current = self.persistence.metadata.get_manifest(dataset_id, version)
            if current != manifest:
                raise GatewayError(409, "dataset_changed")
            if manifest.approval.status == "approved":
                return manifest
            approval = DatasetApproval(
                status="approved",
                actor=identity.subject_id_pseudonymous,
                reason=reason,
                at=now(),
            )
            approved = manifest.model_copy(update={"approval": approval})
            approved = approved.model_copy(
                update={
                    "approval": approval.model_copy(
                        update={"mac": approval_mac(approved, self.persistence.keyring)}
                    )
                }
            )
            self.persistence.metadata.approve_manifest(approved)
            return approved
