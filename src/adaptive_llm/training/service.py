"""Approved, current-policy training; synchronous worker work never runs on the HTTP loop."""

import fcntl
import hashlib
from dataclasses import replace
from pathlib import Path
from time import thread_time

from adaptive_llm.contracts import (
    DatasetLineage,
    DatasetManifest,
    LifecycleTransition,
    ModelManifest,
    TrainingJob,
    TrainingJobSpecification,
    now,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.datasets.builder import DatasetBuilder, LocalDatasetBuilder
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.policy import PolicyEngine
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.registry.artifacts import artifact_digest, signed_metadata, verify_artifact
from adaptive_llm.storage.crypto import PayloadCipher
from adaptive_llm.training import Trainer


class TrainingOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        builder: DatasetBuilder,
        policy: PolicyEngine,
        trainer: Trainer,
        data_dir: Path,
        cipher: PayloadCipher,
        keyring: Keyring,
        revision: str,
    ) -> None:
        self.registry, self.builder, self.policy, self.trainer = registry, builder, policy, trainer
        self.data_dir, self.cipher, self.keyring, self.revision = (
            data_dir,
            cipher,
            keyring,
            revision,
        )

    def _eligible(self, spec: TrainingJobSpecification, identity: Identity) -> DatasetManifest:
        manifest = self.builder.get(spec.dataset_id, spec.dataset_version, identity)
        read_shards(manifest, self.data_dir, self.cipher, self.keyring)
        if manifest.approval.status != "approved":
            raise GatewayError(409, "dataset_approval_required")
        if manifest.purpose != "adapter_training":
            raise GatewayError(409, "training_dataset_required")
        for tenant in manifest.tenant_ids:
            # The application is a server-owned training purpose, never a body-supplied identity.
            trusted = replace(identity, tenant_id=tenant, application_ids=frozenset({"training"}))
            decision = self.policy.decide(trusted, "training")
            if not decision.processing_allowed or not decision.training_allowed:
                raise GatewayError(403, "training_policy_denied")
        return manifest

    def run(self, specification: TrainingJobSpecification, identity: Identity) -> TrainingJob:
        LocalDatasetBuilder._authorize(identity, [])
        if identity.subject_id_pseudonymous is None:
            raise GatewayError(422, "operator_actor_required")
        spec = specification.model_copy(update={"code_revision": self.revision})
        manifest = self._eligible(spec, identity)
        lock_dir = self.data_dir / "control" / "training-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{spec.job_id}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GatewayError(409, "training_job_running") from None
            # Policy may have changed while authenticating artifacts/acquiring the lease.
            manifest = self._eligible(spec, identity)
            return self._run(spec, manifest, identity)

    def _run(
        self, spec: TrainingJobSpecification, dataset: DatasetManifest, identity: Identity
    ) -> TrainingJob:
        job = self.registry.job(spec.job_id, identity)
        if job is not None:
            if job.specification != spec:
                raise GatewayError(409, "training_job_id_conflict")
            if job.state == "succeeded":
                return job
            if job.state == "cancelled":
                raise GatewayError(409, "training_job_cancelled")
        else:
            job = TrainingJob(specification=spec)
            self.registry.save_job(job, dataset.tenant_ids)
        destination = self.data_dir / "models" / spec.registry_id / job.model_version
        working = destination.with_name(f".{job.model_version}.training")
        job = job.model_copy(
            update={
                "state": "running",
                "started_at": job.started_at or now(),
                "completed_at": None,
                "failure_code": None,
            }
        )
        self.registry.save_job(job, dataset.tenant_ids)
        published = False
        try:
            if destination.exists():
                raise GatewayError(409, "model_version_exists")
            working.mkdir(parents=True, exist_ok=True, mode=0o700)

            def checkpoint(reference: str) -> None:
                nonlocal job
                assert job is not None
                job = job.model_copy(update={"checkpoint_refs": [*job.checkpoint_refs, reference]})
                self.registry.save_job(job, dataset.tenant_ids)

            cpu_start = thread_time()
            usage = self.trainer.train(spec, dataset, working, job.checkpoint_refs, checkpoint)
            usage = usage.model_copy(update={"cpu_seconds": thread_time() - cpu_start})
            hashes = {
                p.relative_to(working).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(working.rglob("*"))
                if p.is_file()
            }
            config_digest = hashlib.sha256(
                spec.model_dump_json(exclude={"job_id"}).encode()
            ).hexdigest()
            actor = identity.subject_id_pseudonymous
            if actor is None:
                raise GatewayError(422, "operator_actor_required")
            model = ModelManifest(
                registry_id=spec.registry_id,
                version=job.model_version,
                tenant_ids=sorted(dataset.tenant_ids),
                base_model_id=spec.base_model_id,
                base_model_revision=spec.base_model_revision,
                base_model_licence=spec.base_model_licence,
                adapter_architecture=self.trainer.architecture,
                adapter_config=spec.adapter_config,
                tokenizer_id=spec.tokenizer_id,
                chat_template_version=spec.chat_template_version,
                datasets=[
                    DatasetLineage(
                        dataset_id=dataset.dataset_id,
                        version=dataset.version,
                        content_digest=dataset.content_digest,
                    )
                ],
                training_job_id=spec.job_id,
                code_revision=spec.code_revision,
                container_digest=spec.container_digest,
                configuration_digest=config_digest,
                seed=spec.seed,
                hardware_class=spec.hardware_class,
                artifact_hashes=hashes,
                artifact_digest=artifact_digest(hashes),
                artifact_mac="pending",
                storage_location=destination.relative_to(self.data_dir).as_posix(),
                lifecycle_history=[
                    LifecycleTransition(
                        from_state=None,
                        to_state="candidate",
                        actor=actor,
                        reason="training_completed",
                    )
                ],
            )
            model = model.model_copy(
                update={"artifact_mac": self.keyring.artifact_mac(signed_metadata(model))}
            )
            verify_artifact(model, working, self.keyring)
            working.rename(destination)
            published = True
            job = job.model_copy(
                update={
                    "state": "succeeded",
                    "completed_at": now(),
                    "resource_usage": usage,
                    "artifact_ref": model.storage_location,
                    "artifact_digest": model.artifact_digest,
                }
            )
            self.registry.complete(job, model, trainer_architecture=self.trainer.architecture)
            return job
        except Exception as error:
            if published:
                destination.rename(working)
            permitted_codes = {
                "training_interrupted",
                "checkpoint_integrity_failed",
                "invalid_dataset_artifact",
                "empty_training_split",
                "model_version_exists",
                "artifact_integrity_failed",
                "trainer_architecture_mismatch",
                "single_dataset_required",
            }
            code = (
                error.code
                if isinstance(error, GatewayError) and error.code in permitted_codes
                else "training_failed"
            )
            job = job.model_copy(
                update={
                    "state": "failed",
                    "completed_at": now(),
                    "failure_code": code,
                    "artifact_ref": None,
                    "artifact_digest": None,
                }
            )
            self.registry.save_job(job, dataset.tenant_ids)
            raise GatewayError(503, code) from None
