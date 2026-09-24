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
    excluded = {"artifact_mac", "state", "evaluation_reports", "lifecycle_history"}
    if manifest.manifest_mac_version == "1":
        # Frozen legacy encoding. V2 signs every immutable field, including null/default values.
        if manifest.student_parameter_count is not None or manifest.pruning is not None:
            raise GatewayError(409, "artifact_integrity_failed")
        excluded.update({"manifest_mac_version", "student_parameter_count"})
        if manifest.distillation is None and manifest.student_architecture is None:
            excluded.update({"distillation", "student_architecture"})
        if manifest.capability_signature_version is None:
            excluded.update(
                {
                    "capability_signature_version",
                    "processing_region",
                    "modalities",
                    "tools_supported",
                    "input_micros_per_1000_tokens",
                    "output_micros_per_1000_tokens",
                }
            )
    return json.dumps(
        manifest.model_dump(
            mode="json",
            exclude=excluded,
        ),
        sort_keys=True,
    )


def verify_artifact(manifest: ModelManifest, path: Path, keyring: Keyring) -> dict[str, bytes]:
    """Return authenticated bytes so loaders never reread files after verification."""
    try:
        if (
            manifest.manifest_mac_version == "1"
            and manifest.capability_signature_version is None
            and (
                manifest.processing_region != "local"
                or manifest.modalities != ["text"]
                or manifest.tools_supported
                or manifest.input_micros_per_1000_tokens != 1000
                or manifest.output_micros_per_1000_tokens != 2000
            )
        ):
            raise ValueError
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
