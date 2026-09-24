"""Control-plane registry: state, deployment pointers and events commit together."""

import json
from dataclasses import asdict
from pathlib import Path

from adaptive_llm.contracts import (
    DatasetLineage,
    DeploymentChanged,
    EvaluationReport,
    Event,
    LifecycleTransition,
    ModelManifest,
    PromotionRequest,
    TrainingCompleted,
    TrainingJob,
    now,
    uid,
)
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.storage import EvaluationStore
from adaptive_llm.events.outbox import OutboxStore
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.registry.artifacts import verify_artifact
from adaptive_llm.registry.state import validate_transition
from adaptive_llm.storage.sqlite import SQLiteDatabase


class SQLiteModelRegistry:
    def __init__(
        self,
        database: SQLiteDatabase,
        outbox: OutboxStore,
        keyring: Keyring,
        evaluations: EvaluationStore,
        data_dir: Path,
        pending_limit: int,
    ) -> None:
        self.database, self.outbox, self.keyring = database, outbox, keyring
        self.evaluations, self.data_dir, self.pending_limit = evaluations, data_dir, pending_limit

    def get(self, version: str, identity: Identity) -> ModelManifest:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT data FROM model_versions WHERE version=?", (version,)
            ).fetchone()
        if row is None:
            raise GatewayError(404, "model_not_found")
        manifest = ModelManifest.model_validate_json(row[0])
        if not set(manifest.tenant_ids) <= identity.dataset_tenants:
            raise GatewayError(404, "model_not_found")
        return manifest

    def models(self, identity: Identity) -> list[ModelManifest]:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT data FROM model_versions ORDER BY version"
            ).fetchall()
        models = [ModelManifest.model_validate_json(row[0]) for row in rows]
        return [m for m in models if set(m.tenant_ids) <= identity.dataset_tenants]

    def job(self, job_id: str, identity: Identity) -> TrainingJob | None:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT * FROM training_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        if not set(json.loads(row["tenant_ids"])) <= identity.dataset_tenants:
            raise GatewayError(404, "training_job_not_found")
        return TrainingJob.model_validate_json(row["data"])

    def save_job(self, job: TrainingJob, tenants: list[str]) -> None:
        with self.database.transaction():
            self._save_job_outcome(job, tenants)

    def _save_job_outcome(self, job: TrainingJob, tenants: list[str]) -> None:
        previous = self.database.connection.execute(
            "SELECT data, tenant_ids FROM training_jobs WHERE job_id=?", (job.specification.job_id,)
        ).fetchone()
        if previous is not None:
            tenants = json.loads(previous[1])
            old = TrainingJob.model_validate_json(previous[0])
            if old.state == "cancelled":
                return
            job = job.model_copy(
                update={"cancel_requested": old.cancel_requested or job.cancel_requested}
            )
            if old.state == job.state and old.state in {"failed", "cancelled"}:
                return
        self._save_job(job, tenants)
        if job.state in {"failed", "cancelled"}:
            trace_id = uid()
            self.outbox.enqueue(
                [
                    Event(
                        event_type="training.completed.v1",
                        producer="training_orchestrator",
                        tenant_id=tenant,
                        trace_id=trace_id,
                        data=TrainingCompleted(
                            job_id=job.specification.job_id,
                            job_type="adapter",
                            dataset_refs=[
                                f"{job.specification.dataset_id}/{job.specification.dataset_version}"
                            ],
                            model_version=None,
                            status="failed" if job.state == "failed" else "cancelled",
                            failure_code=job.failure_code,
                        ),
                    )
                    for tenant in sorted(tenants)
                ],
                self.pending_limit,
            )

    def enqueue(self, job: TrainingJob, tenants: list[str], identity: Identity) -> TrainingJob:
        trusted = asdict(identity)
        trusted["application_ids"] = sorted(identity.application_ids)
        trusted["dataset_tenants"] = sorted(identity.dataset_tenants)
        with self.database.transaction():
            old = self.job(job.specification.job_id, identity)
            if old is not None:
                if old.trainer_architecture != job.trainer_architecture:
                    raise GatewayError(409, "training_trainer_mismatch")
                if old.specification != job.specification:
                    raise GatewayError(409, "training_job_id_conflict")
                if old.state != "failed":
                    if old.state in {"queued", "running"}:
                        # Legacy 3a jobs acquire a trusted submitter on authenticated retry.
                        self.database.connection.execute(
                            "INSERT OR IGNORE INTO training_submitters VALUES (?, ?)",
                            (job.specification.job_id, json.dumps(trusted)),
                        )
                    return old
                job = old.model_copy(
                    update={"state": "queued", "failure_code": None, "completed_at": None}
                )
            self._save_job(job, tenants)
            self.database.connection.execute(
                "INSERT INTO training_submitters VALUES (?, ?) ON CONFLICT(job_id) "
                "DO UPDATE SET identity=excluded.identity",
                (job.specification.job_id, json.dumps(trusted)),
            )
            return job

    def pending_jobs(self) -> list[tuple[TrainingJob, Identity]]:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT j.data, s.identity FROM training_jobs j JOIN training_submitters s "
                "USING(job_id) WHERE json_extract(j.data, '$.state') IN ('queued', 'running') "
                "ORDER BY json_extract(j.data, '$.created_at'), j.job_id"
            ).fetchall()
        result = []
        for row in rows:
            trusted = json.loads(row[1])
            trusted["application_ids"] = frozenset(trusted["application_ids"])
            trusted["dataset_tenants"] = frozenset(trusted["dataset_tenants"])
            result.append((TrainingJob.model_validate_json(row[0]), Identity(**trusted)))
        return result

    def cancel_job(self, job_id: str, identity: Identity) -> TrainingJob:
        with self.database.transaction():
            job = self.job(job_id, identity)
            if job is None:
                raise GatewayError(404, "training_job_not_found")
            if job.state in {"succeeded", "failed", "cancelled"}:
                return job
            job = job.model_copy(update={"cancel_requested": True})
            if job.state == "queued":
                job = job.model_copy(update={"state": "cancelled", "completed_at": now()})
            row = self.database.connection.execute(
                "SELECT tenant_ids FROM training_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            self._save_job_outcome(job, json.loads(row[0]))
            return job

    def record_evaluation(self, report: EvaluationReport, identity: Identity) -> None:
        """Publication hook runs inside the evaluation store's control transaction."""
        if not self.database.connection.in_transaction:
            raise GatewayError(503, "control_transaction_required")
        row = self.database.connection.execute(
            "SELECT data FROM model_versions WHERE version=?",
            (report.specification.candidate_deployment_id,),
        ).fetchone()
        if row is None:
            return
        model = self.get(report.specification.candidate_deployment_id, identity)
        if (
            model.state in {"deprecated", "revoked"}
            or report.candidate_artifact_digest != model.artifact_digest
        ):
            raise GatewayError(409, "evaluation_model_changed")
        updated = model.model_copy(
            update={
                "evaluation_reports": {
                    **model.evaluation_reports,
                    report.specification.evaluation_id: report.passed,
                }
            }
        )
        self.database.connection.execute(
            "UPDATE model_versions SET data=? WHERE version=?",
            (updated.model_dump_json(), model.version),
        )

    def _save_job(self, job: TrainingJob, tenants: list[str]) -> None:
        self.database.connection.execute(
            "INSERT INTO training_jobs VALUES (?, ?, ?) ON CONFLICT(job_id) "
            "DO UPDATE SET data=excluded.data",
            (job.specification.job_id, json.dumps(sorted(tenants)), job.model_dump_json()),
        )

    def complete(
        self, job: TrainingJob, manifest: ModelManifest, *, trainer_architecture: str
    ) -> None:
        if manifest.adapter_architecture != trainer_architecture:
            raise GatewayError(409, "trainer_architecture_mismatch")
        self._dataset(manifest)
        with self.database.transaction():
            current = self.database.connection.execute(
                "SELECT data FROM training_jobs WHERE job_id=?", (job.specification.job_id,)
            ).fetchone()
            if current and TrainingJob.model_validate_json(current[0]).cancel_requested:
                raise GatewayError(409, "training_cancelled")
            self.database.connection.execute(
                "INSERT INTO model_versions VALUES (?, ?, ?, ?)",
                (
                    manifest.version,
                    manifest.registry_id,
                    json.dumps(sorted(manifest.tenant_ids)),
                    manifest.model_dump_json(),
                ),
            )
            for transition in manifest.lifecycle_history:
                self._record(manifest, transition)
            self._save_job(job, manifest.tenant_ids)
            trace_id = uid()
            self.outbox.enqueue(
                [
                    Event(
                        event_type="training.completed.v1",
                        producer="training_orchestrator",
                        tenant_id=tenant,
                        trace_id=trace_id,
                        data=TrainingCompleted(
                            job_id=job.specification.job_id,
                            job_type="adapter",
                            dataset_refs=[f"{d.dataset_id}/{d.version}" for d in manifest.datasets],
                            model_version=manifest.version,
                            status="succeeded",
                            artifact_digest=manifest.artifact_digest,
                        ),
                    )
                    for tenant in sorted(manifest.tenant_ids)
                ],
                self.pending_limit,
            )

    @staticmethod
    def _dataset(manifest: ModelManifest) -> DatasetLineage:
        if len(manifest.datasets) != 1:
            raise GatewayError(409, "single_dataset_required")
        return manifest.datasets[0]

    @staticmethod
    def _approval(manifest: ModelManifest) -> LifecycleTransition | None:
        return next(
            (
                transition
                for transition in reversed(manifest.lifecycle_history)
                if transition.from_state == "evaluating" and transition.to_state == "approved"
            ),
            None,
        )

    def _passed(
        self, manifest: ModelManifest, evaluation_id: str | None, identity: Identity
    ) -> bool:
        dataset = self._dataset(manifest)
        if evaluation_id is None:
            return False
        report = self.evaluations.get(evaluation_id, identity)
        if report is None or report.specification.baseline_deployment_id is None:
            return False
        baseline = self.evaluations.baseline(
            report.specification.baseline_deployment_id, dataset.version, identity
        )
        return bool(
            baseline
            and baseline.passed
            and report.passed
            and all(g.passed for g in decisions(report))
            and report.specification.candidate_deployment_id == manifest.version
            and report.candidate_manifest_version == manifest.version
            and report.candidate_artifact_digest == manifest.artifact_digest
            and report.specification.dataset_id == dataset.dataset_id
            and report.specification.dataset_version == dataset.version
            and report.dataset_content_digest == dataset.content_digest
            and report.baseline_report_id == baseline.specification.evaluation_id
            and report.baseline_manifest_version == baseline.candidate_manifest_version
            and baseline.specification.dataset_id == dataset.dataset_id
        )

    def _record(self, manifest: ModelManifest, transition: LifecycleTransition) -> None:
        self.database.connection.execute(
            "INSERT INTO lifecycle_transitions VALUES (?, ?, ?)",
            (uid(), manifest.version, transition.model_dump_json()),
        )
        trace_id = uid()
        self.outbox.enqueue(
            [
                Event(
                    event_type="deployment.changed.v1",
                    producer="deployment_controller",
                    tenant_id=tenant,
                    trace_id=trace_id,
                    data=DeploymentChanged(
                        deployment_id=manifest.registry_id,
                        model_version=manifest.version,
                        previous_state=transition.from_state,
                        new_state=transition.to_state,
                        actor_id=transition.actor,
                        reason=transition.reason,
                        evaluation_id=transition.evaluation_id,
                    ),
                )
                for tenant in sorted(manifest.tenant_ids)
            ],
            self.pending_limit,
        )

    def _transition(
        self,
        manifest: ModelManifest,
        request: PromotionRequest,
        identity: Identity,
        *,
        rollback: bool = False,
    ) -> ModelManifest:
        # Recovery trusts recorded approval, independent of replaceable evaluation state.
        passed = (
            self._approval(manifest) is not None
            if rollback
            else self._passed(manifest, request.evaluation_id, identity)
        )
        validate_transition(
            manifest.state,
            request.target_state,
            actor=identity.subject_id_pseudonymous,
            reason=request.reason,
            evaluation_passed=passed,
            rollback=rollback,
        )
        if request.target_state not in {"revoked", "deprecated"}:
            verify_artifact(manifest, self.data_dir / manifest.storage_location, self.keyring)
        assert identity.subject_id_pseudonymous is not None
        transition = LifecycleTransition(
            from_state=manifest.state,
            to_state=request.target_state,
            actor=identity.subject_id_pseudonymous,
            reason=request.reason,
            evaluation_id=request.evaluation_id,
        )
        reports = dict(manifest.evaluation_reports)
        if request.evaluation_id is not None and passed and not rollback:
            reports[request.evaluation_id] = passed
        updated = manifest.model_copy(
            update={
                "state": request.target_state,
                "lifecycle_history": [*manifest.lifecycle_history, transition],
                "evaluation_reports": reports,
            }
        )
        self.database.connection.execute(
            "UPDATE model_versions SET data=? WHERE version=?",
            (updated.model_dump_json(), manifest.version),
        )
        self._record(updated, transition)
        return updated

    def promote(self, request: PromotionRequest, identity: Identity) -> ModelManifest:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.transaction():
            manifest = self.get(request.model_version, identity)
            if request.actor is not None and request.actor != identity.subject_id_pseudonymous:
                raise GatewayError(403, "actor_mismatch")
            if manifest.state == request.target_state and manifest.lifecycle_history:
                last = manifest.lifecycle_history[-1]
                if (
                    last.actor == identity.subject_id_pseudonymous
                    and last.reason == request.reason
                    and last.evaluation_id == request.evaluation_id
                ):
                    return manifest
            updated = self._transition(manifest, request, identity)
            if updated.state == "production":
                row = self.database.connection.execute(
                    "SELECT * FROM deployments WHERE deployment_id=?", (manifest.registry_id,)
                ).fetchone()
                previous = row["current_version"] if row else None
                if previous:
                    incumbent = self.get(previous, identity)
                    if (
                        incumbent.state != "production"
                        or incumbent.tenant_ids != manifest.tenant_ids
                    ):
                        raise GatewayError(409, "deployment_conflict")
                    self._transition(
                        incumbent,
                        PromotionRequest(
                            model_version=previous, target_state="deprecated", reason=request.reason
                        ),
                        identity,
                    )
                self.database.connection.execute(
                    "INSERT INTO deployments VALUES (?, ?, ?) ON CONFLICT(deployment_id) "
                    "DO UPDATE SET "
                    "current_version=excluded.current_version, "
                    "previous_version=excluded.previous_version",
                    (manifest.registry_id, manifest.version, previous),
                )
            return updated

    def rollback(self, deployment_id: str, identity: Identity, reason: str) -> list[ModelManifest]:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.transaction():
            row = self.database.connection.execute(
                "SELECT * FROM deployments WHERE deployment_id=?", (deployment_id,)
            ).fetchone()
            if row is None or row["previous_version"] is None:
                raise GatewayError(409, "previous_version_required")
            current, previous = (
                self.get(row[key], identity) for key in ("current_version", "previous_version")
            )
            approval = self._approval(previous)
            if current.state != "production" or previous.state != "deprecated" or approval is None:
                raise GatewayError(409, "rollback_version_unavailable")
            retired = self._transition(
                current,
                PromotionRequest(
                    model_version=current.version, target_state="deprecated", reason=reason
                ),
                identity,
                rollback=True,
            )
            restored = self._transition(
                previous,
                PromotionRequest(
                    model_version=previous.version,
                    target_state="production",
                    reason=reason,
                    evaluation_id=approval.evaluation_id,
                ),
                identity,
                rollback=True,
            )
            self.database.connection.execute(
                "UPDATE deployments SET current_version=?, previous_version=? "
                "WHERE deployment_id=?",
                (previous.version, current.version, deployment_id),
            )
            return [retired, restored]
