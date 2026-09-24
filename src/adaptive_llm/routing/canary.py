"""Content-free live cohorts, segmented matched bootstrap and persistent automatic rollback."""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import fmean
from threading import Event
from time import monotonic
from typing import Protocol

from adaptive_llm.contracts import (
    CanaryAggregate,
    CanaryReport,
    LiveObservation,
    OutcomeAggregate,
    RollbackThresholds,
    RoutePolicy,
    now,
)
from adaptive_llm.evaluation.stats import paired_bootstrap, percentile
from adaptive_llm.gateway.identity import Identity
from adaptive_llm.routing.control import RouteSnapshot


class LiveOutcomes(Protocol):
    def record_live(self, observation: LiveObservation) -> None: ...


class CanaryStore(Protocol):
    def snapshot(self) -> RouteSnapshot: ...
    def canary_report(
        self, policy_id: str, since: datetime, identity: Identity
    ) -> CanaryReport: ...
    def automatic_disable(
        self, policy: RoutePolicy, version: str, aggregate: CanaryAggregate, identity: Identity
    ) -> None: ...


def outcomes(rows: list[LiveObservation]) -> OutcomeAggregate:
    count = len(rows)
    successes = sum(r.success for r in rows)
    cost = sum(r.total_cost_micros for r in rows)
    return OutcomeAggregate(
        interactions=count,
        successes=successes,
        total_cost_micros=cost,
        cost_per_success_micros=(cost + successes - 1) // successes if successes else None,
        validation_failure_rate=fmean(r.validation_failure for r in rows) if rows else None,
        error_rate=fmean(r.error for r in rows) if rows else None,
        p95_latency_ms=percentile([r.latency_ms for r in rows], 0.95) if rows else None,
        critical_safety_incidents=sum(r.critical_safety_incidents for r in rows),
    )


def reduction(candidate: OutcomeAggregate, foundation: OutcomeAggregate) -> float | None:
    # Ratios use exact totals/denominators, never the rounded display values.
    if not candidate.successes or not foundation.successes or not foundation.total_cost_micros:
        return None
    return 1 - (candidate.total_cost_micros * foundation.successes) / (
        foundation.total_cost_micros * candidate.successes
    )


def matched(
    left: list[LiveObservation],
    right: list[LiveObservation],
    *,
    costs: bool,
) -> tuple[list[float], list[float]]:
    groups: dict[tuple[str, str, str, str, bool], list[list[LiveObservation]]] = defaultdict(
        lambda: [[], []]
    )
    for arm, rows in enumerate((left, right)):
        for row in rows:
            f = row.features
            groups[row.tenant_id, f.task, f.risk_tier, f.language, f.context_supplied][arm].append(
                row
            )
    a: list[float] = []
    b: list[float] = []
    for arms in groups.values():
        values: list[list[float]] = []
        for rows in arms:
            ordered = sorted(rows, key=lambda r: (r.created_at, r.interaction_id))
            if costs:
                successes = [r for r in ordered if r.success]
                failed_cost = sum(r.total_cost_micros for r in ordered if not r.success)
                values.append(
                    [r.total_cost_micros + failed_cost / len(successes) for r in successes]
                )
            else:
                values.append([r.quality for r in ordered])
        for x, y in zip(*values, strict=False):
            a.append(x)
            b.append(y)
    return a, b


def aggregate(rows: list[LiveObservation], limits: RollbackThresholds) -> CanaryAggregate:
    # Attempt cohorts retain failed specialist and fallback costs even when foundation served.
    specialist = [r for r in rows if r.specialist_version is not None]
    foundation = [r for r in rows if r.specialist_version is None]
    served = [r for r in rows if r.specialist_served]
    foundation_served = [r for r in rows if r.success and not r.specialist_served]
    s, f, ss, fs = map(outcomes, (specialist, foundation, served, foundation_served))
    quality = paired_bootstrap(*matched(served, foundation, costs=False))
    cost = paired_bootstrap(*matched(specialist, foundation, costs=True))
    savings = reduction(s, f)
    reasons: list[str] = []
    if s.critical_safety_incidents:
        reasons.append("critical_safety_incident")
    if s.interactions >= limits.minimum_samples:
        for name, value, maximum in (
            (
                "validation_failure_rate",
                s.validation_failure_rate,
                limits.validation_failure_rate_max,
            ),
            ("error_rate", s.error_rate, limits.error_rate_max),
            ("p95_latency_ms", s.p95_latency_ms, limits.p95_latency_ms_max),
            ("cost_ratio", 1 - savings if savings is not None else None, limits.cost_ratio_max),
        ):
            if value is not None and value > maximum:
                reasons.append(name)
    enough = (
        quality.sample_size >= limits.minimum_samples and cost.sample_size >= limits.minimum_samples
    )
    return CanaryAggregate(
        specialist=s,
        foundation=f,
        specialist_served=ss,
        foundation_served=fs,
        quality_delta=quality,
        cost_delta_micros=cost,
        served_cost_delta_micros=paired_bootstrap(*matched(served, foundation_served, costs=True)),
        cost_reduction_fraction=savings,
        served_cost_reduction_fraction=reduction(ss, fs),
        shadow_cost_micros=sum(r.shadow_cost_micros for r in rows),
        passed=bool(
            enough
            and not reasons
            and quality.ci_lower is not None
            and quality.ci_lower > -limits.non_inferiority_margin
            and savings is not None
            and 1 - savings <= limits.cost_ratio_max
        ),
        breach_reasons=reasons,
    )


class CanaryMonitor:
    def __init__(self, store: CanaryStore, identity: Identity) -> None:
        self.store, self.identity = store, identity
        self.stopping = Event()
        self.failures = 0

    def work_once(self) -> None:
        snapshot = self.store.snapshot()
        policy = snapshot.policy
        if policy is None or not policy.live_specialists_allowed or snapshot.killed:
            return
        report = self.store.canary_report(
            policy.policy_id,
            now() - timedelta(seconds=policy.rollback.window_seconds),
            self.identity,
        )
        for version, measurement in report.specialists.items():
            if version not in snapshot.disabled_specialists and measurement.breach_reasons:
                self.store.automatic_disable(policy, version, measurement, self.identity)

    async def run(self) -> None:
        due = 0.0
        while not self.stopping.is_set():
            if monotonic() >= due:
                try:
                    await asyncio.to_thread(self.work_once)
                    snapshot = await asyncio.to_thread(self.store.snapshot)
                    interval = snapshot.policy.rollback.interval_seconds if snapshot.policy else 1
                except Exception:
                    self.failures += 1
                    interval = 1
                due = monotonic() + interval
            await asyncio.sleep(0.05)

    def stop(self) -> None:
        self.stopping.set()
