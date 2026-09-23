"""Purpose-separated MACs authenticate the full artifact inventory and immutable lineage."""

import hashlib
import hmac
import json
from pathlib import Path

from adaptive_llm.contracts import ModelManifest
from adaptive_llm.gateway.identity import GatewayError, Keyring


def artifact_digest(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def signed_metadata(manifest: ModelManifest) -> str:
    return json.dumps(
        manifest.model_dump(
            mode="json",
            exclude={"artifact_mac", "state", "evaluation_reports", "lifecycle_history"},
        ),
        sort_keys=True,
    )


def verify_artifact(manifest: ModelManifest, path: Path, keyring: Keyring) -> dict[str, bytes]:
    """Return authenticated bytes so loaders never reread files after verification."""
    try:
        if not hmac.compare_digest(
            manifest.artifact_mac, keyring.artifact_mac(signed_metadata(manifest))
        ):
            raise ValueError
        if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError
        files = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        if files != set(manifest.artifact_hashes):
            raise ValueError
        verified: dict[str, bytes] = {}
        for name, digest in manifest.artifact_hashes.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError
            content = (path / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError
            verified[name] = content
        if manifest.artifact_digest != artifact_digest(manifest.artifact_hashes):
            raise ValueError
        return verified
    except Exception:
        raise GatewayError(409, "artifact_integrity_failed") from None
