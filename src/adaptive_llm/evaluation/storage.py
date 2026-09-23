"""Transactional report, baseline and outbox publication with detached local MACs."""

import hmac
import json
import shutil
from pathlib import Path
from typing import Protocol

from adaptive_llm.contracts import EvaluationCompleted, EvaluationReport, Event, uid
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.events.outbox import OutboxStore
from adaptive_llm.gateway.identity import GatewayError, Identity, Keyring
from adaptive_llm.storage.sqlite import SQLiteDatabase, timestamp


class EvaluationStore(Protocol):
    def get(self, evaluation_id: str, identity: Identity) -> EvaluationReport | None: ...

    def baseline(
        self, deployment_id: str, dataset_version: str, identity: Identity
    ) -> EvaluationReport | None: ...

    def publish(
        self,
        report: EvaluationReport,
        tenants: list[str],
        identity: Identity,
        *,
        lock_baseline: bool,
        expected_baseline: str | None,
        note: str | None,
    ) -> None: ...


class SQLiteEvaluationStore:
    def __init__(
        self,
        database: SQLiteDatabase,
        outbox: OutboxStore,
        keyring: Keyring,
        data_dir: Path,
        pending_limit: int,
    ) -> None:
        self.database, self.outbox, self.keyring = database, outbox, keyring
        self.data_dir, self.pending_limit = data_dir, pending_limit

    def get(self, evaluation_id: str, identity: Identity) -> EvaluationReport | None:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT * FROM evaluation_reports WHERE evaluation_id = ?", (evaluation_id,)
            ).fetchone()
        if row is None or not set(json.loads(row["tenant_ids"])) <= identity.dataset_tenants:
            return None
        try:
            # Use the committed id, never the URL path, to construct an artifact path.
            report = EvaluationReport.model_validate_json(row["report"])
            directory = self.data_dir / "evaluations" / report.specification.evaluation_id
            encoded = (directory / "report.json").read_text()
            mac = self.keyring.report_mac(encoded)
            if (
                encoded != row["report"]
                or not hmac.compare_digest(mac, row["report_mac"])
                or not hmac.compare_digest(mac, (directory / "report.mac").read_text())
            ):
                raise ValueError
            return report
        except Exception:
            raise GatewayError(503, "evaluation_integrity_failed") from None

    def baseline(
        self, deployment_id: str, dataset_version: str, identity: Identity
    ) -> EvaluationReport | None:
        LocalDatasetBuilder._authorize(identity, [])
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT evaluation_id FROM baselines WHERE deployment_id=? AND dataset_version=?",
                (deployment_id, dataset_version),
            ).fetchone()
        return self.get(row["evaluation_id"], identity) if row else None

    def publish(
        self,
        report: EvaluationReport,
        tenants: list[str],
        identity: Identity,
        *,
        lock_baseline: bool,
        expected_baseline: str | None,
        note: str | None,
    ) -> None:
        LocalDatasetBuilder._authorize(identity, tenants)
        spec = report.specification
        destination = self.data_dir / "evaluations" / spec.evaluation_id
        staging = destination.with_name(f".{spec.evaluation_id}.{uid()}.building")
        encoded = report.model_dump_json(indent=2)
        mac = self.keyring.report_mac(encoded)
        created = published = False
        try:
            staging.mkdir(parents=True, mode=0o700)
            created = True
            (staging / "report.json").write_text(encoded)
            (staging / "report.mac").write_text(mac)
            trace_id = uid()
            events = [
                Event(
                    event_type="evaluation.completed.v1",
                    producer="evaluation_service",
                    tenant_id=tenant,
                    trace_id=trace_id,
                    data=EvaluationCompleted(
                        evaluation_id=spec.evaluation_id,
                        model_version=report.candidate_manifest_version,
                        baseline_version=report.baseline_manifest_version,
                        suites=[suite.suite for suite in report.suite_results],
                        passed=report.passed,
                        report_ref=f"evaluations/{spec.evaluation_id}/report.json",
                    ),
                )
                for tenant in sorted(tenants)
            ]
            with self.database.transaction():
                connection = self.database.connection
                if (
                    connection.execute(
                        "SELECT 1 FROM evaluation_reports WHERE evaluation_id=?",
                        (spec.evaluation_id,),
                    ).fetchone()
                    or destination.exists()
                ):
                    raise GatewayError(409, "evaluation_exists")
                baseline = connection.execute(
                    "SELECT evaluation_id FROM baselines "
                    "WHERE deployment_id=? AND dataset_version=?",
                    (spec.baseline_deployment_id, spec.dataset_version),
                ).fetchone()
                if (baseline["evaluation_id"] if baseline else None) != expected_baseline:
                    raise GatewayError(409, "baseline_changed")
                connection.execute(
                    "INSERT INTO evaluation_reports VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        spec.evaluation_id,
                        json.dumps(sorted(tenants)),
                        encoded,
                        mac,
                        self.keyring.fingerprint(note) if note else None,
                        identity.subject_id_pseudonymous,
                        timestamp(report.completed_at),
                    ),
                )
                if lock_baseline:
                    connection.execute(
                        "INSERT INTO baselines VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(deployment_id, dataset_version) DO UPDATE SET "
                        "evaluation_id=excluded.evaluation_id, "
                        "manifest_version=excluded.manifest_version, "
                        "locked_at=excluded.locked_at",
                        (
                            spec.candidate_deployment_id,
                            spec.dataset_version,
                            spec.dataset_id,
                            spec.evaluation_id,
                            report.candidate_manifest_version,
                            timestamp(report.completed_at),
                        ),
                    )
                self.outbox.enqueue(events, self.pending_limit)
                staging.rename(destination)
                published = True
        except BaseException:
            if created:
                shutil.rmtree(staging, ignore_errors=True)
            if published:
                shutil.rmtree(destination, ignore_errors=True)
            raise
