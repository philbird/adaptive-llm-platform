"""Immutable policy versions, independent activation and persistent disablement."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from threading import Lock
from time import monotonic
from typing import Literal, Protocol

from adaptive_llm.contracts import (
    DeploymentChanged,
    Environment,
    Event,
    RouteControlNote,
    RoutePolicy,
    ShadowAggregate,
    ShadowComparison,
    ShadowReport,
    now,
    uid,
)
from adaptive_llm.datasets.builder import LocalDatasetBuilder
from adaptive_llm.evaluation.stats import paired_bootstrap
from adaptive_llm.events.outbox import OutboxStore
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.storage.sqlite import SQLiteDatabase


@dataclass(frozen=True)
class RouteSnapshot:
    policy: RoutePolicy | None = None
    killed: bool = False
    disabled_tenants: frozenset[str] = frozenset()
    disabled_tasks: frozenset[str] = frozenset()

    def enabled(self, tenant: str, task: str) -> bool:
        p = self.policy
        return bool(
            p
            and not self.killed
            and not p.kill_switch
            and tenant not in self.disabled_tenants
            and task not in self.disabled_tasks
            and p.tenant_enabled.get(tenant, False)
            and p.task_enabled.get(task, False)
        )


class RoutePolicyStore(Protocol):
    def snapshot(self) -> RouteSnapshot: ...
    def observe(
        self, policy_id: str, interaction_id: str, tenant: str, segments: list[str]
    ) -> None: ...
    def compare(self, comparison: ShadowComparison) -> None: ...


class SQLiteRoutePolicies:
    def __init__(
        self,
        database: SQLiteDatabase,
        outbox: OutboxStore,
        registry: ModelRegistry,
        environment: Environment,
        foundation_id: str,
        tenants: frozenset[str],
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.database, self.outbox, self.registry = database, outbox, registry
        self.environment, self.foundation_id, self.tenants = environment, foundation_id, tenants
        self.clock = clock
        self._lock = Lock()
        self._cached = RouteSnapshot()
        self._expires = 0.0

    def _authorize(self, identity: Identity, tenants: list[str]) -> None:
        LocalDatasetBuilder._authorize(identity, tenants)
        if identity.environment != self.environment or identity.subject_id_pseudonymous is None:
            raise GatewayError(403, "environment_forbidden")

    def _validate(self, policy: RoutePolicy, identity: Identity) -> None:
        self._authorize(identity, list(policy.tenant_enabled))
        if policy.foundation_fallback != self.foundation_id:
            raise GatewayError(422, "foundation_fallback_unavailable")
        if not set(policy.tenant_enabled) <= self.tenants:
            raise GatewayError(422, "route_tenant_unavailable")
        for version in policy.eligible_specialist_versions:
            model = self.registry.get(version, identity)
            if model.state != "shadow":
                raise GatewayError(409, "shadow_state_required")

    def create(self, policy: RoutePolicy, identity: Identity) -> RoutePolicy:
        with self.database.transaction():
            self._validate(policy, identity)
            if self.database.connection.execute(
                "SELECT 1 FROM route_policies WHERE policy_id=?", (policy.policy_id,)
            ).fetchone():
                raise GatewayError(409, "immutable_route_policy")
            self.database.connection.execute(
                "INSERT INTO route_policies VALUES (?, ?, ?, ?, ?, ?)",
                (
                    policy.policy_id,
                    self.environment,
                    json.dumps(sorted(policy.tenant_enabled)),
                    now().isoformat(),
                    identity.subject_id_pseudonymous,
                    policy.model_dump_json(),
                ),
            )
        return policy

    def get(self, policy_id: str, identity: Identity) -> RoutePolicy:
        self._authorize(identity, [])
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT data, tenant_ids FROM route_policies WHERE policy_id=? AND environment=?",
                (policy_id, self.environment),
            ).fetchone()
        if row is None or not set(json.loads(row[1])) <= identity.dataset_tenants:
            raise GatewayError(404, "route_policy_not_found")
        return RoutePolicy.model_validate_json(row[0])

    def _ensure_pointer(self) -> None:
        self.database.connection.execute(
            "INSERT OR IGNORE INTO route_policy_active(environment) VALUES (?)", (self.environment,)
        )

    def activate(self, policy_id: str, identity: Identity, note: RouteControlNote) -> RoutePolicy:
        if note.tenant_id is not None or note.task is not None:
            raise GatewayError(422, "activation_scope_invalid")
        # Pointer replacement affects the environment, so require all its tenant grants.
        self._authorize(identity, sorted(self.tenants))
        with self.database.transaction():
            policy = self.get(policy_id, identity)
            self._validate(policy, identity)
            self._ensure_pointer()
            self.database.connection.execute(
                "UPDATE route_policy_active SET policy_id=? WHERE environment=?",
                (policy_id, self.environment),
            )
            self._audit(identity, note, policy_id, "activate", "active")
        self._invalidate()
        return policy

    def switch(self, disabled: bool, identity: Identity, note: RouteControlNote) -> RouteSnapshot:
        self._authorize(identity, [note.tenant_id] if note.tenant_id else sorted(self.tenants))
        if note.tenant_id is not None and note.tenant_id not in self.tenants:
            raise GatewayError(422, "route_tenant_unavailable")
        with self.database.transaction():
            self._ensure_pointer()
            row = self.database.connection.execute(
                "SELECT * FROM route_policy_active WHERE environment=?", (self.environment,)
            ).fetchone()
            if note.tenant_id is not None or note.task is not None:
                column = "disabled_tenants" if note.tenant_id is not None else "disabled_tasks"
                values = set(json.loads(row[column]))
                value = note.tenant_id or note.task
                if disabled:
                    values.add(value)
                else:
                    values.discard(value)
                if len(values) > 100:
                    raise GatewayError(422, "disablement_limit")
                self.database.connection.execute(
                    f"UPDATE route_policy_active SET {column}=? WHERE environment=?",
                    (json.dumps(sorted(values)), self.environment),
                )
            else:
                self.database.connection.execute(
                    "UPDATE route_policy_active SET kill_switch=? WHERE environment=?",
                    (int(disabled), self.environment),
                )
            self._audit(
                identity,
                note,
                row["policy_id"],
                "disable" if disabled else "enable",
                "disabled" if disabled else "enabled",
            )
        self._invalidate()
        return self.snapshot()

    def _audit(
        self,
        identity: Identity,
        note: RouteControlNote,
        policy_id: str | None,
        action: str,
        state: Literal["active", "enabled", "disabled"],
    ) -> None:
        assert identity.subject_id_pseudonymous is not None
        scope = note.tenant_id or note.task or "environment"
        self.database.connection.execute(
            "INSERT INTO route_policy_history VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                uid(),
                self.environment,
                policy_id,
                identity.subject_id_pseudonymous,
                note.reason,
                f"{action}:{scope}",
                now().isoformat(),
            ),
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
                        deployment_id="route_policy",
                        model_version=policy_id or "none",
                        previous_state=None,
                        new_state=state,
                        actor_id=identity.subject_id_pseudonymous,
                        reason=note.reason,
                    ),
                )
                for tenant in ([note.tenant_id] if note.tenant_id else sorted(self.tenants))
            ],
            100_000,
        )

    def _invalidate(self) -> None:
        with self._lock:
            self._expires = 0

    def snapshot(self) -> RouteSnapshot:
        with self._lock:
            if self.clock() < self._expires:
                return self._cached
            with self.database.lock:
                row = self.database.connection.execute(
                    "SELECT a.*, p.data FROM route_policy_active a LEFT JOIN route_policies p "
                    "ON p.policy_id=a.policy_id WHERE a.environment=?",
                    (self.environment,),
                ).fetchone()
            self._cached = RouteSnapshot(
                policy=RoutePolicy.model_validate_json(row["data"])
                if row and row["data"]
                else None,
                killed=bool(row["kill_switch"]) if row else False,
                disabled_tenants=frozenset(json.loads(row["disabled_tenants"]))
                if row
                else frozenset(),
                disabled_tasks=frozenset(json.loads(row["disabled_tasks"])) if row else frozenset(),
            )
            self._expires = self.clock() + 1
            return self._cached

    def observe(
        self, policy_id: str, interaction_id: str, tenant: str, segments: list[str]
    ) -> None:
        with self.database.transaction():
            self.database.connection.execute(
                "INSERT OR IGNORE INTO shadow_observations VALUES (?, ?, ?, ?, ?)",
                (interaction_id, policy_id, tenant, now().isoformat(), json.dumps(segments)),
            )

    def compare(self, comparison: ShadowComparison) -> None:
        with self.database.transaction():
            self.database.connection.execute(
                "INSERT OR IGNORE INTO shadow_comparisons VALUES (?, ?)",
                (comparison.interaction_id, comparison.model_dump_json()),
            )

    def report(self, policy_id: str, since: datetime, identity: Identity) -> ShadowReport:
        self.get(policy_id, identity)
        if since.tzinfo is None:
            raise GatewayError(422, "utc_since_required")
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT o.tenant_id, o.segments, c.data FROM shadow_observations o "
                "LEFT JOIN shadow_comparisons c USING(interaction_id) "
                "WHERE o.policy_id=? AND o.created_at>=? ORDER BY o.interaction_id",
                (policy_id, since.astimezone(UTC).isoformat()),
            ).fetchall()
        selected = [r for r in rows if r[0] in identity.dataset_tenants]
        comparisons = [ShadowComparison.model_validate_json(r[2]) for r in selected if r[2]]
        segments = sorted({s for r in selected for s in json.loads(r[1])})
        return ShadowReport(
            policy_id=policy_id,
            since=since,
            overall=aggregate(comparisons, len(selected)),
            critical_segments={
                segment: aggregate(
                    [c for c in comparisons if segment in c.segments],
                    sum(segment in json.loads(r[1]) for r in selected),
                )
                for segment in segments
            },
        )


def aggregate(rows: list[ShadowComparison], opportunities: int) -> ShadowAggregate:
    return ShadowAggregate(
        opportunities=opportunities,
        comparisons=len(rows),
        coverage=len(rows) / opportunities if opportunities else 0,
        specialist_validation_pass_rate=fmean(c.specialist_validation.passed for c in rows)
        if rows
        else None,
        score_delta=paired_bootstrap(
            [c.specialist_score for c in rows], [c.foundation_score for c in rows]
        ),
        mean_cost_delta_micros=fmean(c.cost_delta_micros for c in rows) if rows else None,
        mean_latency_delta_ms=fmean(c.latency_delta_ms for c in rows) if rows else None,
        mean_input_token_delta=fmean(c.input_token_delta for c in rows) if rows else None,
        mean_output_token_delta=fmean(c.output_token_delta for c in rows) if rows else None,
        citation_metrics={
            name: fmean(getattr(c, name) for c in rows)
            for name in (
                "foundation_citation_precision",
                "foundation_citation_recall",
                "specialist_citation_precision",
                "specialist_citation_recall",
            )
        }
        if rows
        else {},
    )
