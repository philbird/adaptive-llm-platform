"""Deterministic, content-free fake artifacts, with authenticated restartable checkpoints."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from adaptive_llm.contracts import DatasetManifest, ResourceUsage, TrainingJobSpecification
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.gateway.identity import GatewayError, Keyring
from adaptive_llm.storage.crypto import PayloadCipher


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise GatewayError(409, "checkpoint_integrity_failed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


class FakeTrainer:
    architecture = "deterministic-fake-adapter-v1"

    def __init__(
        self,
        data_dir: Path,
        cipher: PayloadCipher,
        keyring: Keyring,
        *,
        fail_after_checkpoint: int | None = None,
    ) -> None:
        self.data_dir, self.cipher, self.keyring = data_dir, cipher, keyring
        self.fail_after_checkpoint = fail_after_checkpoint
        self.executed_steps: list[int] = []

    def train(
        self,
        specification: TrainingJobSpecification,
        dataset: DatasetManifest,
        directory: Path,
        checkpoint_refs: list[str],
        checkpoint: Callable[[str], None],
        check: Callable[[], None],
    ) -> ResourceUsage:
        rows = read_shards(dataset, self.data_dir, self.cipher, self.keyring)["train"]
        if not rows:
            raise GatewayError(409, "empty_training_split")
        config = specification.adapter_config.model_dump_json().encode()
        # User-derived examples are keyed before deriving public artifact bytes.
        examples_digest = self.keyring.dataset_hash(b"\n".join(rows).decode())
        digest = hashlib.sha256(
            examples_digest.encode() + config + str(specification.seed).encode()
        ).digest()
        for step in (1, 2):
            check()
            name = f"checkpoint-{step}"
            if name in checkpoint_refs and not all(
                (directory / name / filename).is_file()
                for filename in ("adapter_weights.bin", "training_report.json", "checkpoint.mac")
            ):
                raise GatewayError(409, "checkpoint_integrity_failed")
            report = json.dumps(
                {
                    "steps": step,
                    "examples": len(rows),
                    "seed": specification.seed,
                    "loss_curve": [
                        round((1 + specification.seed % 97) / (len(rows) + i), 8)
                        for i in range(1, step + 1)
                    ],
                    "weights_digest": hashlib.sha256(digest).hexdigest(),
                },
                sort_keys=True,
            ).encode()
            checkpoint_weights = hashlib.sha256(digest + bytes([step])).digest()
            write_once(directory / name / "adapter_weights.bin", checkpoint_weights)
            write_once(directory / name / "training_report.json", report)
            signed = hashlib.sha256(checkpoint_weights + report).hexdigest()
            write_once(
                directory / name / "checkpoint.mac", self.keyring.artifact_mac(signed).encode()
            )
            if name not in checkpoint_refs:
                self.executed_steps.append(step)
                checkpoint(name)
                check()
                if self.fail_after_checkpoint == step:
                    self.fail_after_checkpoint = None
                    raise GatewayError(503, "training_interrupted")
            # The next step starts from the verified checkpoint, including after restart.
            digest = (directory / name / "adapter_weights.bin").read_bytes()
        write_once(directory / "adapter_config.json", config)
        write_once(directory / "adapter_weights.bin", digest)
        write_once(
            directory / "training_report.json",
            (directory / "checkpoint-2/training_report.json").read_bytes(),
        )
        return ResourceUsage(
            examples=len(rows),
            steps=2,
            artifact_bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
        )
