"""Approved, current-policy training; synchronous worker work never runs on the HTTP loop."""

import asyncio
import fcntl
import hashlib
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import perf_counter, thread_time

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
    @staticmethod
    def _move_checkpoints(source: Path, destination: Path) -> None:
        """Separate resumable state from exports; also recover partially completed moves."""
        if any(p.is_symlink() for p in (source, source.parent, destination, destination.parent)):
            raise GatewayError(409, "checkpoint_integrity_failed")
        checkpoints = sorted([*source.glob("checkpoint-*"), *source.glob(".checkpoint-*")])
        for path in checkpoints:
            if path.is_symlink() or not path.is_dir() or (destination / path.name).exists():
                raise GatewayError(409, "checkpoint_integrity_failed")
        if checkpoints:
            destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            for path in checkpoints:
                path.rename(destination / path.name)

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
        self.stopping = Event()

    def submit(self, specification: TrainingJobSpecification, identity: Identity) -> TrainingJob:
        LocalDatasetBuilder._authorize(identity, [])
        if identity.subject_id_pseudonymous is None:
            raise GatewayError(422, "operator_actor_required")
        spec = specification.model_copy(update={"code_revision": self.revision})
        dataset = self._eligible(spec, identity)
        return self.registry.enqueue(
            TrainingJob(specification=spec, trainer_architecture=self.trainer.architecture),
            dataset.tenant_ids,
            identity,
        )

    def stop(self) -> None:
        self.stopping.set()

    async def worker(self) -> None:
        while not self.stopping.is_set():
            await asyncio.to_thread(self.work_once)
            await asyncio.sleep(0.05)

    def work_once(self) -> None:
        # A process-wide lease also serializes separate local app/CLI processes.
        lock_dir = self.data_dir / "control" / "training-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            for job, identity in self.registry.pending_jobs():
                if self.stopping.is_set():
                    return
                if job.trainer_architecture != self.trainer.architecture:
                    continue
                try:
                    self.run(job.specification, identity)
                except GatewayError as error:
                    if error.code == "training_shutdown":
                        return
                    current = self.registry.job(job.specification.job_id, identity)
                    if current is not None and current.state in {"queued", "running"}:
                        if error.code == "training_job_running":
                            return
                        # Eligibility/revision failures happen before the trainer's failure handler.
                        self.registry.save_job(
                            current.model_copy(
                                update={
                                    "state": "failed",
                                    "completed_at": now(),
                                    "failure_code": "training_eligibility_failed",
                                }
                            ),
                            sorted(identity.dataset_tenants),
                        )
                return

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
            if job.trainer_architecture != self.trainer.architecture:
                raise GatewayError(409, "training_trainer_mismatch")
            if job.specification != spec:
                raise GatewayError(409, "training_job_id_conflict")
            if job.state == "succeeded":
                return job
            if job.state == "cancelled":
                return job
        else:
            job = TrainingJob(specification=spec, trainer_architecture=self.trainer.architecture)
            self.registry.save_job(job, dataset.tenant_ids)
        destination = self.data_dir / "models" / spec.registry_id / job.model_version
        working = destination.with_name(f".{job.model_version}.training")
        archived_checkpoints = destination.parent / ".checkpoints" / job.model_version
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
            # A publication failure or crash may leave all or part of the checkpoints archived.
            self._move_checkpoints(archived_checkpoints, working)

            def checkpoint(reference: str) -> None:
                nonlocal job
                assert job is not None
                job = job.model_copy(
                    update={
                        "checkpoint_refs": list(dict.fromkeys([*job.checkpoint_refs, reference]))
                    }
                )
                self.registry.save_job(job, dataset.tenant_ids)

            def check() -> None:
                current = self.registry.job(spec.job_id, identity)
                if current is not None and current.cancel_requested:
                    raise GatewayError(409, "training_cancelled")
                if self.stopping.is_set():
                    raise GatewayError(503, "training_shutdown")

            cpu_start = thread_time()
            wall_start = perf_counter()
            usage = self.trainer.train(
                spec, dataset, working, job.checkpoint_refs, checkpoint, check
            )
            check()
            usage = usage.model_copy(
                update={
                    "cpu_seconds": thread_time() - cpu_start,
                    "wall_seconds": perf_counter() - wall_start,
                }
            )
            self._move_checkpoints(working, archived_checkpoints)
            hashes = {
                p.relative_to(working).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(working.rglob("*"))
                if p.is_file()
            }
            usage = usage.model_copy(
                update={
                    "artifact_bytes": sum((working / name).stat().st_size for name in hashes),
                }
            )
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
                context_limit=spec.max_sequence_length,
                safety_notes=[
                    "Local synthetic lifecycle verification; quality requires evaluation."
                ],
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
                "training_cancelled",
                "training_shutdown",
                "training_memory_limit",
                "training_dependencies_unavailable",
                "invalid_base_model",
                "training_configuration_invalid",
            }
            code = (
                error.code
                if isinstance(error, GatewayError) and error.code in permitted_codes
                else "training_failed"
            )
            job = job.model_copy(
                update={
                    "state": "cancelled"
                    if code == "training_cancelled"
                    else "queued"
                    if code == "training_shutdown"
                    else "failed",
                    "completed_at": None if code == "training_shutdown" else now(),
                    "failure_code": None if code == "training_shutdown" else code,
                    "artifact_ref": None,
                    "artifact_digest": None,
                }
            )
            self.registry.save_job(job, dataset.tenant_ids)
            raise GatewayError(503, code) from None
