"""MAC-authenticated study inventories; aggregate tensors are AES-GCM ciphertext only."""

import hashlib
import hmac
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol
from uuid import UUID

from adaptive_llm.contracts import now
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.research.models import StudySummary
from adaptive_llm.storage import EncryptedPayload
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.training.lora import encoded


def authorize(identity: Identity) -> None:
    LocalDatasetBuilder._authorize(identity, [])
    if "research" not in identity.capabilities:
        raise GatewayError(403, "research_capability_required")
    if identity.environment != "local":
        raise GatewayError(403, "research_local_only")


class StudyStore(Protocol):
    def get(self, study_id: str, identity: Identity) -> tuple[StudySummary, bytes]: ...
    def save(self, summary: StudySummary, tensors: bytes, identity: Identity) -> None: ...


def markdown(summary: StudySummary) -> str:
    return (
        "# Local activation research\n\n```json\n" + summary.model_dump_json(indent=2) + "\n```\n"
    )


class LocalStudyStore:
    def __init__(self, root: Path, cipher: PayloadCipher, keyring: Keyring) -> None:
        self.root, self.cipher, self.keyring = root, cipher, keyring

    def path(self, study_id: str) -> Path:
        try:
            parsed = UUID(study_id)
            if parsed.version != 7 or str(parsed) != study_id:
                raise ValueError
            path = self.root / "studies" / study_id
            if any(p.is_symlink() for p in (self.root, path.parent, path)):
                raise ValueError
            return path
        except Exception:
            raise GatewayError(404, "study_not_found") from None

    def _mac(self, envelope: dict[str, Any]) -> str:
        return self.keyring.artifact_mac("research-study-v1:" + encoded(envelope).decode())

    def save(self, summary: StudySummary, tensors: bytes, identity: Identity) -> None:
        authorize(identity)
        LocalDatasetBuilder._authorize(identity, summary.tenant_ids)
        path = self.path(summary.specification.study_id)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = self.cipher.encrypt(
            tensors,
            "research",
            summary.specification.study_id,
            "aggregates",
            now(),
            aad_field=f"research/{summary.specification.study_id}",
        )
        envelope: dict[str, Any] = {
            "summary": summary.model_dump(mode="json"),
            "nonce": payload.nonce.hex(),
            "key_version": payload.key_version,
            "digest": hashlib.sha256(payload.ciphertext).hexdigest(),
        }
        with TemporaryDirectory(prefix=".study-", dir=self.root) as temporary:
            staging = Path(temporary) / "complete"
            staging.mkdir(mode=0o700)
            (staging / "aggregates.safetensors.enc").write_bytes(payload.ciphertext)
            (staging / "report.md").write_text(markdown(summary))
            (staging / "summary.json").write_bytes(
                encoded({**envelope, "mac": self._mac(envelope)})
            )
            if path.exists():
                raise GatewayError(409, "study_id_conflict")
            staging.rename(path)

    def get(self, study_id: str, identity: Identity) -> tuple[StudySummary, bytes]:
        authorize(identity)
        path = self.path(study_id)
        if not path.exists():
            raise GatewayError(404, "study_not_found")
        try:
            if any(p.is_symlink() for p in path.rglob("*")):
                raise ValueError
            envelope = json.loads((path / "summary.json").read_bytes())
            mac = envelope.pop("mac")
            if not hmac.compare_digest(mac, self._mac(envelope)):
                raise ValueError
            summary = StudySummary.model_validate(envelope["summary"])
            LocalDatasetBuilder._authorize(identity, summary.tenant_ids)
            if summary.specification.study_id != study_id:
                raise ValueError
            ciphertext = (path / "aggregates.safetensors.enc").read_bytes()
            if hashlib.sha256(ciphertext).hexdigest() != envelope["digest"]:
                raise ValueError
            payload = EncryptedPayload(
                reference=study_id,
                tenant_id="research",
                interaction_id=study_id,
                field="aggregates",
                nonce=bytes.fromhex(envelope["nonce"]),
                ciphertext=ciphertext,
                key_version=envelope["key_version"],
                expires_at=now(),
            )
            raw = self.cipher.decrypt(
                payload,
                "research",
                study_id,
                "aggregates",
                aad_field=f"research/{study_id}",
            )
            return summary, raw
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(409, "study_integrity_failed") from None
