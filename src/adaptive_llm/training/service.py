"""Approved, current-policy training; synchronous worker work never runs on the HTTP loop."""

import asyncio
import fcntl
import hashlib
import json
from collections.abc import Callable
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
from adaptive_llm.signing import sign_record
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
        *,
        student_memory_limit_bytes: int = 2_000_000_000,
        student_time_limit_seconds: float = 300,
    ) -> None:
        self.registry, self.builder, self.policy, self.trainer = registry, builder, policy, trainer
        self.data_dir, self.cipher, self.keyring, self.revision = (
            data_dir,
            cipher,
            keyring,
            revision,
        )
        self.stopping = Event()
        self.distillation_eligibility: Callable[[DatasetManifest, Identity], None] | None = None
        from adaptive_llm.routing.train import RouterTrainer

        self.router_trainer = RouterTrainer(data_dir, cipher, keyring)
        from adaptive_llm.distillation.training import StudentTrainer

        self.student_trainers = {
            mode: StudentTrainer(
                data_dir,
                cipher,
                keyring,
                full=mode == "full",
                memory_limit_bytes=student_memory_limit_bytes,
                time_limit_seconds=student_time_limit_seconds,
            )
            for mode in ("full", "lora")
        }

    def _trainer(self, spec: TrainingJobSpecification) -> Trainer:
        if spec.job_type == "distillation":
            return self.student_trainers[spec.student_training]
        return self.router_trainer if spec.job_type == "router" else self.trainer

    def submit(self, specification: TrainingJobSpecification, identity: Identity) -> TrainingJob:
        LocalDatasetBuilder._authorize(identity, [])
        if identity.subject_id_pseudonymous is None:
            raise GatewayError(422, "operator_actor_required")
        spec = specification.model_copy(update={"code_revision": self.revision})
        dataset = self._eligible(spec, identity)
        return self.registry.enqueue(
            TrainingJob(specification=spec, trainer_architecture=self._trainer(spec).architecture),
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
        with self.registry.worker_claim(
            (
                self.trainer.architecture,
                self.router_trainer.architecture,
                *(t.architecture for t in self.student_trainers.values()),
            )
        ) as jobs:
            for job, identity in jobs:
                if self.stopping.is_set():
                    return
                if job.trainer_architecture != self._trainer(job.specification).architecture:
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
        if manifest.purpose != (
            "router_training"
            if spec.job_type == "router"
            else "distillation"
            if spec.job_type == "distillation"
            else "adapter_training"
        ):
            raise GatewayError(409, "training_dataset_required")
        if spec.job_type == "distillation":
            if manifest.distillation is None:
                raise GatewayError(409, "distillation_lineage_required")
            if self.distillation_eligibility is None:
                raise GatewayError(503, "distillation_unavailable")
            self.distillation_eligibility(manifest, identity)
            source = self.builder.get(
                manifest.distillation.source_dataset_id,
                manifest.distillation.source_dataset_version,
                identity,
            )
            read_shards(source, self.data_dir, self.cipher, self.keyring)
            if (
                source.approval.status != "approved"
                or source.purpose != "adapter_training"
                or source.content_digest != manifest.distillation.source_content_digest
            ):
                raise GatewayError(409, "approved_adapter_source_required")
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
        trainer = self._trainer(spec)
        job = self.registry.job(spec.job_id, identity)
        if job is not None:
            if job.trainer_architecture != trainer.architecture:
                raise GatewayError(409, "training_trainer_mismatch")
            if job.specification != spec:
                raise GatewayError(409, "training_job_id_conflict")
            if job.state == "succeeded":
                return job
            if job.state == "cancelled":
                return job
        else:
            job = TrainingJob(specification=spec, trainer_architecture=trainer.architecture)
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
            usage = trainer.train(spec, dataset, working, job.checkpoint_refs, checkpoint, check)
            check()
            if spec.job_type == "distillation":
                self._eligible(spec, identity)
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
            student_architecture: str | None = None
            student_parameter_count: int | None = None
            limitations = [
                "Local synthetic training; production quality requires separate acceptance."
            ]
            if spec.job_type == "distillation":
                # This report was produced from the same verified snapshot used for training.
                report = json.loads((working / "training_report.json").read_bytes())
                student_architecture = report["student_architecture"]
                student_parameter_count = report["student_parameter_count"]
                if (
                    dataset.distillation is not None
                    and dataset.distillation.teacher_parameter_count is None
                ):
                    limitations.append("teacher size unknown; size reduction not verified")
            model = ModelManifest(
                registry_id=spec.registry_id,
                version=job.model_version,
                tenant_ids=sorted(dataset.tenant_ids),
                base_model_id=spec.base_model_id,
                base_model_revision=spec.base_model_revision,
                base_model_licence=spec.base_model_licence,
                adapter_architecture=trainer.architecture,
                adapter_config=spec.adapter_config,
                tokenizer_id=spec.tokenizer_id,
                chat_template_version=spec.chat_template_version,
                context_limit=spec.max_sequence_length,
                capability_signature_version="1",
                input_micros_per_1000_tokens=spec.input_micros_per_1000_tokens,
                output_micros_per_1000_tokens=spec.output_micros_per_1000_tokens,
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
                manifest_mac_version="2",
                storage_location=destination.relative_to(self.data_dir).as_posix(),
                lifecycle_history=[
                    LifecycleTransition(
                        from_state=None,
                        to_state="candidate",
                        actor=actor,
                        reason="training_completed",
                    )
                ],
                student_architecture=student_architecture,
                student_parameter_count=student_parameter_count,
                known_limitations=limitations,
                distillation=dataset.distillation if spec.job_type == "distillation" else None,
            )
            self.keyring.require_signer()
            if self.keyring.signer is not None:
                model = model.model_copy(update={"artifact_mac": ""})
                model = sign_record(
                    model, self.keyring.signer, "model-manifest", signed_metadata(model)
                )
            else:
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
            self.registry.complete(job, model, trainer_architecture=trainer.architecture)
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
                "student_must_be_smaller",
                "soft_target_tokenizer_mismatch",
                "invalid_soft_targets",
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
