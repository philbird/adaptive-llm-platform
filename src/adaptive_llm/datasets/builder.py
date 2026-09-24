"""Snapshot, compute without storage locks, then publish immutable local datasets."""

import base64
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from adaptive_llm.contracts import (
    DatasetBuilt,
    DatasetManifest,
    DatasetQualitySummary,
    DatasetSpecification,
    DistillationLineage,
    Event,
    now,
    uid,
)
from adaptive_llm.datasets.construction import BuiltExample, Constructor, Excluded, PayloadSnapshot
from adaptive_llm.datasets.curation import SPLITS, curate, golden_texts, split_examples
from adaptive_llm.datasets.eligibility import Example, Selection, select
from adaptive_llm.datasets.sources import SourceResolver
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.signing import FIELDS, sign_record
from adaptive_llm.storage import EncryptedPayload, StorageError
from adaptive_llm.storage.persistence import Persistence


class DatasetBuilder(Protocol):
    def build(self, specification: DatasetSpecification, identity: Identity) -> DatasetManifest: ...

    def get(self, dataset_id: str, version: str, identity: Identity) -> DatasetManifest: ...


class RoutingExamples(Protocol):
    def build(
        self, examples: list[Example], specification: DatasetSpecification, identity: Identity
    ) -> list[BuiltExample]: ...


@dataclass
class DistilledExamples:
    examples: list[BuiltExample]
    lineage: DistillationLineage
    soft_targets: dict[str, bytes]


class DistillationExamples(Protocol):
    def build(
        self,
        examples: list[Example],
        specification: DatasetSpecification,
        identity: Identity,
        exclusions: Counter[str],
    ) -> DistilledExamples: ...

    def validate(self, specification: DatasetSpecification, identity: Identity) -> None: ...


def code_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return (
            result
            if len(result) == 40 and all(c in "0123456789abcdef" for c in result)
            else "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class LocalDatasetBuilder:
    def __init__(
        self,
        persistence: Persistence,
        policy: PolicyEngine,
        sources: SourceResolver,
        data_dir: Path,
        golden_dir: Path,
        *,
        revision: str,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.persistence, self.policy = persistence, policy
        self.data_dir, self.golden_dir, self.clock = data_dir, golden_dir, clock
        self.revision = revision
        self.routing_examples: RoutingExamples | None = None
        self.distillation_examples: DistillationExamples | None = None
        self.constructor = Constructor(
            persistence.payloads,
            persistence.cipher,
            persistence.keyring,
            persistence.redactor,
            sources,
        )

    @staticmethod
    def _authorize(identity: Identity, tenants: list[str]) -> None:
        if identity.key_class != "operator":
            raise GatewayError(403, "operator_required")
        if not set(tenants) <= identity.dataset_tenants:
            raise GatewayError(403, "dataset_tenant_forbidden")

    def get(self, dataset_id: str, version: str, identity: Identity) -> DatasetManifest:
        self._authorize(identity, [])
        manifest = self.persistence.metadata.get_manifest(dataset_id, version)
        if manifest is None or not set(manifest.tenant_ids) <= identity.dataset_tenants:
            raise GatewayError(404, "dataset_not_found")
        return manifest

    def _snapshot(
        self,
        specification: DatasetSpecification,
    ) -> tuple[datetime, Selection, PayloadSnapshot]:
        blobs: dict[tuple[str, str], EncryptedPayload] = {}
        with self.persistence.metadata.read_transaction():
            at = self.clock()
            selection = select(self.persistence.metadata, self.policy, specification, at)
            if specification.purpose in {"router_training", "distillation"}:
                return at, selection, PayloadSnapshot(blobs)
            for candidate in selection.examples:
                tenant = candidate.interaction.tenant_id
                refs = {candidate.interaction.input.messages_ref}
                if candidate.attempt is not None:
                    refs.add(candidate.attempt.output_ref)
                refs.update(f.correction_ref for f in candidate.feedback if f.training_authorised)
                for ref in refs:
                    if ref is not None:
                        blob = self.persistence.payloads.get(tenant, ref, at)
                        if blob is not None:
                            blobs[tenant, ref] = blob
        return at, selection, PayloadSnapshot(blobs)

    def _deleted(self, candidates: list[Example]) -> set[tuple[str, str]]:
        """Called under the publish lock; these candidates had no tombstones at snapshot."""
        deleted: set[tuple[str, str]] = set()
        metadata = self.persistence.metadata
        for candidate in candidates:
            interaction = candidate.interaction
            tenant, iid = interaction.tenant_id, interaction.interaction_id
            subject = interaction.subject_id_pseudonymous
            if metadata.get_tombstone(tenant, iid) is not None or (
                subject is not None and metadata.get_subject_tombstone(tenant, subject) is not None
            ):
                deleted.add((tenant, iid))
        return deleted

    def build(self, specification: DatasetSpecification, identity: Identity) -> DatasetManifest:
        self._authorize(identity, specification.tenant_ids)
        # Phase 1: copy metadata, feedback and encrypted blobs, then release the read lock.
        at, selection, payloads = self._snapshot(specification)
        version, trace_id = uid(), uid()
        destination = self.data_dir / "datasets" / specification.dataset_id / version
        staging = destination.with_name(f".{version}.building")
        examples: list[BuiltExample] = []
        exclusions = selection.exclusions.copy()
        distilled: DistilledExamples | None = None
        # Phase 2: no database reads or locks during decryption, construction or curation.
        for candidate in (
            selection.examples
            if specification.purpose not in {"router_training", "distillation"}
            else []
        ):
            if candidate.interaction.environment != identity.environment:
                exclusions["environment_forbidden"] += 1
                continue
            try:
                examples.append(
                    self.constructor.build(candidate, specification, at, payloads=payloads)
                )
            except Excluded as error:
                exclusions[str(error)] += 1
        if specification.purpose == "router_training":
            if self.routing_examples is None:
                raise GatewayError(422, "routing_observations_unavailable")
            examples = self.routing_examples.build(
                [
                    e
                    for e in selection.examples
                    if e.interaction.environment == identity.environment
                ],
                specification,
                identity,
            )
        if specification.purpose == "distillation":
            if self.distillation_examples is None:
                raise GatewayError(422, "distillation_unavailable")
            distilled = self.distillation_examples.build(
                [
                    e
                    for e in selection.examples
                    if e.interaction.environment == identity.environment
                ],
                specification,
                identity,
                exclusions,
            )
            examples = distilled.examples
        golden = golden_texts(self.golden_dir)
        candidates = selection.examples
        created = False
        published = False
        try:
            staging.mkdir(parents=True, exist_ok=False, mode=0o700)
            created = True
            while True:
                manifest = self._stage(
                    specification,
                    version,
                    at,
                    selection.considered,
                    examples,
                    exclusions,
                    golden,
                    staging,
                    distilled,
                )
                digest = hashlib.sha256(manifest.model_dump_json(indent=2).encode()).hexdigest()
                events = [
                    Event(
                        event_type="dataset.built.v1",
                        producer="dataset_builder",
                        tenant_id=tenant,
                        trace_id=trace_id,
                        data=DatasetBuilt(
                            dataset_id=manifest.dataset_id,
                            version=version,
                            purpose=manifest.purpose,
                            manifest_digest=digest,
                            deletions_applied_through=at,
                        ),
                    )
                    for tenant in sorted(specification.tenant_ids)
                ]
                # Phase 3: only tombstone reads, metadata/event writes and one directory rename.
                with self.persistence.metadata.transaction():
                    if distilled is not None and self.distillation_examples is not None:
                        self.distillation_examples.validate(specification, identity)
                    deleted = self._deleted(candidates)
                    if not deleted:
                        if destination.exists():
                            raise StorageError("dataset_version_exists")
                        self.persistence.metadata.put_manifest(manifest)
                        self.persistence.outbox.enqueue(
                            events, self.persistence.outbox_pending_limit
                        )
                        staging.rename(destination)
                        published = True
                if not deleted:
                    break
                # Regenerate outside the transaction and re-check before publishing. Re-curating
                # also promotes a surviving duplicate when its earlier exemplar was deleted.
                exclusions["deleted_during_build"] += len(deleted)
                candidates = [
                    c
                    for c in candidates
                    if (c.interaction.tenant_id, c.interaction.interaction_id) not in deleted
                ]
                examples = [e for e in examples if (e.tenant_id, e.interaction_id) not in deleted]
        except BaseException:
            if created:
                shutil.rmtree(staging, ignore_errors=True)
            if published:
                shutil.rmtree(destination, ignore_errors=True)
            raise
        self.persistence.refresh_metrics()
        return manifest

    def _stage(
        self,
        specification: DatasetSpecification,
        version: str,
        at: datetime,
        considered: int,
        examples: list[BuiltExample],
        exclusions: Counter[str],
        golden: list[str],
        staging: Path,
        distilled: DistilledExamples | None = None,
    ) -> DatasetManifest:
        counts = exclusions.copy()
        accepted = (
            examples
            if specification.purpose in {"router_training", "distillation"}
            else curate(examples, golden, specification.near_duplicate_threshold, counts)
        )
        splits = (
            {s: [e for e in accepted if e.row["split"] == s] for s in SPLITS}
            if distilled is not None
            else split_examples(accepted, specification)
        )
        if len(splits["train"]) < specification.minimum_examples:
            raise GatewayError(422, "insufficient_training_examples")
        plaintext_hashes: list[str] = []
        # Stable tenant/split order, independent of encryption nonces and build version.
        for tenant in sorted(specification.tenant_ids):
            for split in SPLITS:
                rows = [
                    dict(example.row, split=split, transformation_code_revision=self.revision)
                    for example in splits[split]
                    if example.tenant_id == tenant
                ]
                plaintext = "".join(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                    for row in rows
                ).encode()
                digest = hashlib.sha256(plaintext).hexdigest()
                plaintext_hashes.append(digest)
                blob = self.persistence.cipher.encrypt(
                    plaintext,
                    tenant,
                    f"{specification.dataset_id}/{version}",
                    "dataset",
                    # Generic cipher expiry is not serialized; artifact lifecycle is separate.
                    at,
                    aad_field=split,
                )
                envelope = {
                    "field": "dataset",
                    "tenant_id": tenant,
                    "split": split,
                    "key_version": blob.key_version,
                    "nonce": base64.b64encode(blob.nonce).decode(),
                    "ciphertext": base64.b64encode(blob.ciphertext).decode(),
                    "plaintext_hash": digest,
                }
                (staging / f"{tenant}.{split}.jsonl.enc").write_text(json.dumps(envelope))
        lineage = None
        if distilled is not None:
            from adaptive_llm.contracts import SoftTargetFile

            # Publication retries after a deletion must not retain orphan tensor payloads.
            for stale in staging.glob("*.safetensors.enc"):
                stale.unlink()
            soft_files = []
            for example in splits["train"]:
                content = distilled.soft_targets.get(example.exact_hash)
                if content is None:
                    continue
                name = f"{example.exact_hash}.safetensors.enc"
                blob = self.persistence.cipher.encrypt(
                    content,
                    example.tenant_id,
                    f"{specification.dataset_id}/{version}",
                    "dataset",
                    at,
                    aad_field=name,
                )
                (staging / name).write_bytes(blob.nonce + blob.ciphertext)
                soft_files.append(
                    SoftTargetFile(
                        tenant_id=example.tenant_id,
                        example_hash=example.exact_hash,
                        plaintext_hash=hashlib.sha256(content).hexdigest(),
                        key_version=blob.key_version,
                    )
                )
            lineage = distilled.lineage.model_copy(
                update={
                    "soft_target_files": soft_files,
                    "mix_counts": dict(
                        Counter(str(e.row["distillation_kind"]) for e in splits["train"])
                    ),
                }
            )
            plaintext_hashes.extend(f.plaintext_hash for f in soft_files)
            considered += sum(e.row.get("distillation_kind") != "teacher" for e in splits["train"])
        manifest = DatasetManifest(
            dataset_id=specification.dataset_id,
            version=version,
            purpose=specification.purpose,
            tenant_ids=sorted(specification.tenant_ids),
            created_at=self.clock(),
            source_window=specification.source_window,
            eligibility_policy_version=specification.eligibility_policy_version,
            transformation_code_revision=self.revision,
            redaction_version=self.persistence.redactor.version,
            examples={split: len(splits[split]) for split in SPLITS},
            split_strategy=specification.split_strategy,
            content_digest=hashlib.sha256("".join(plaintext_hashes).encode()).hexdigest(),
            deletions_applied_through=at,
            quality_summary=DatasetQualitySummary(
                considered=considered,
                accepted_rate=len(accepted) / max(1, considered),
                duplicate_rate=(counts["exact_duplicate"] + counts["near_duplicate"])
                / max(1, considered),
                exclusions=dict(sorted(counts.items())),
                label_mix=dict(sorted(Counter(e.target_source for e in accepted).items())),
                languages=dict(sorted(Counter(e.language for e in accepted).items())),
            ),
            specification=specification,
            distillation=lineage,
        )
        self.persistence.keyring.require_signer()
        if self.persistence.keyring.signer is not None:
            manifest = sign_record(
                manifest,
                self.persistence.keyring.signer,
                "dataset-manifest",
                manifest.model_dump_json(exclude=FIELDS | {"approval"}),
            )
        encoded = manifest.model_dump_json(indent=2)
        (staging / "manifest.json").write_text(encoded)
        (staging / "manifest.mac").write_text(
            "" if manifest.signature else self.persistence.keyring.manifest_mac(encoded)
        )
        (staging / "data-card.md").write_text(self._data_card(manifest))
        return manifest

    @staticmethod
    def _data_card(manifest: DatasetManifest) -> str:
        quality = manifest.quality_summary
        return (
            f"# Local synthetic dataset {manifest.dataset_id}\n\n"
            f"Version: {manifest.version}\n\nApproval: pending\n\n"
            f"Split counts: {json.dumps(manifest.examples, sort_keys=True)}\n\n"
            f"Exclusions: {json.dumps(quality.exclusions, sort_keys=True)}\n\n"
            f"Target label mix: {json.dumps(quality.label_mix, sort_keys=True)}\n\n"
            "Integrity: versioned Ed25519 signature; legacy records retain their local MAC.\n\n"
            "Limitations: synthetic data; placeholder system/policy instructions; no tool results; "
            "lexical 5-gram decontamination only; local regex redaction; no human verification; "
            "difficulty unknown; no balancing; group sizes may skew 80/10/10 splits. "
            "Time-crossing groups move wholly to their latest partition. Missing payloads and "
            "unavailable exact source versions are excluded. No training approval is granted. "
            "Future builds apply deletions; existing immutable versions are not rewritten.\n"
        )
