"""Independent hard gates; lower cost cannot compensate for failed quality or safety."""

import json
from pathlib import Path

from adaptive_llm.contracts import EvaluationReport, GateDecision, PairedComparison

REQUIRED_SUITES = {"golden", "held_out", "safety", "retrieval", "performance"}


def decisions(report: EvaluationReport) -> list[GateDecision]:
    if report.specification.suites == ["routing"]:
        return routing_decisions(report)
    spec = report.specification
    suites = {suite.suite: suite for suite in report.suite_results}
    baseline_suites = {suite.suite: suite for suite in report.baseline_suite_results}
    gates: list[GateDecision] = []

    def add(gate: str, passed: bool, reason: str) -> None:
        gates.append(GateDecision(gate=gate, passed=passed, reason="met" if passed else reason))

    add(
        "required_suites",
        set(suites) == REQUIRED_SUITES
        and set(spec.suites) == REQUIRED_SUITES
        and set(baseline_suites) == REQUIRED_SUITES
        and report.baseline_manifest_version is not None
        and len(suites) == len(report.suite_results)
        and all(s.completed and s.items > 0 for s in [*suites.values(), *baseline_suites.values()]),
        "missing_suite_or_items",
    )
    minimum = max(spec.minimum_sample_size or 0, report.derived_minimum_sample_size)

    def enough(comparison: PairedComparison) -> bool:
        return spec.minimum_sample_size is not None and comparison.sample_size >= minimum

    def noninferior(comparison: PairedComparison) -> bool:
        return (
            enough(comparison)
            and comparison.ci_lower is not None
            and comparison.ci_upper is not None
            and comparison.mean_delta is not None
            and comparison.ci_lower > -spec.non_inferiority_margin
        )

    add("sample_size", enough(report.paired_comparison), "insufficient_samples")
    add("non_inferiority", noninferior(report.paired_comparison), "quality_or_coverage_failed")
    if report.distillation is not None:
        comparison = report.segment_comparisons.get("distillation", PairedComparison(sample_size=0))
        add(
            "distillation_non_inferiority",
            noninferior(comparison),
            "teacher_quality_or_coverage_failed",
        )
        for key in spec.critical_segments:
            comparisons = [
                value
                for name, value in report.segment_comparisons.items()
                if name.startswith(f"distillation.{key}.")
            ]
            add(
                f"distillation.{key}",
                bool(comparisons)
                and sum(c.sample_size for c in comparisons) == comparison.sample_size
                and all(noninferior(c) for c in comparisons),
                "teacher_segment_failed",
            )
    for key in spec.critical_segments:
        segments = {
            name: value
            for name, value in report.segment_comparisons.items()
            if name.startswith(f"{key}.")
        }
        add(
            f"segment.{key}",
            bool(segments)
            and sum(value.sample_size for value in segments.values())
            == report.paired_comparison.sample_size
            and all(noninferior(value) for value in segments.values()),
            "segment_quality_or_coverage_failed",
        )
    for name in ("golden", "safety", "retrieval"):
        suite = suites.get(name)
        add(
            f"{name}_assertions",
            suite is not None and not any(suite.failures.values()),
            "deterministic_or_critical_failure",
        )
    safety = suites.get("safety")
    retrieval = suites.get("retrieval")
    add(
        "safety_privacy_isolation",
        safety is not None
        and retrieval is not None
        and all(
            safety.metrics.get(key) == 0
            for key in ("critical_failures", "leakage_rate", "cross_tenant_incidents")
        )
        and retrieval.metrics.get("acl_incidents") == 0,
        "critical_regression",
    )
    performance = suites.get("performance")
    for gate, metric, maximum in (
        ("latency", "p95_latency_ms", spec.latency_p95_ms_max),
        ("error_rate", "error_rate", spec.error_rate_max),
        ("cost", "cost_per_success_micros", spec.cost_per_success_micros_max),
    ):
        value = (
            performance.cost_per_success_micros
            if performance and gate == "cost"
            else performance.metrics.get(metric)
            if performance
            else None
        )
        add(
            gate,
            value is not None
            and value <= maximum
            and performance is not None
            and performance.metrics.get("concurrency") == spec.concurrency
            and performance.items == spec.performance_requests,
            "target_or_measurement_failed",
        )
    add(
        "report_completeness",
        bool(report.known_limitations)
        and bool(report.coverage)
        and all(value > 0 for value in report.coverage.values())
        and report.paired_comparison.ci_lower is not None
        and report.paired_comparison.ci_upper is not None,
        "missing_ci_coverage_or_limitations",
    )
    return gates


def routing_decisions(report: EvaluationReport) -> list[GateDecision]:
    targets = json.loads(
        (
            Path(__file__).resolve().parents[3] / "configs/evaluation/initial-targets.json"
        ).read_text()
    )
    suite = report.suite_results[0] if len(report.suite_results) == 1 else None
    enough = bool(
        suite
        and suite.suite == "routing"
        and suite.completed
        and suite.items
        >= max(
            report.specification.minimum_sample_size or 0,
            targets["minimum_sample_sizes"]["overall"],
        )
    )
    return [
        GateDecision(gate=name, passed=passed, reason="met" if passed else "routing_gate_failed")
        for name, passed in (
            ("routing_samples", enough),
            (
                "false_specialist_rate",
                bool(
                    suite
                    and suite.metrics.get("false_specialist_rate", 1)
                    <= targets["false_specialist_rate_max"]
                ),
            ),
            ("calibration_error", bool(suite and suite.metrics.get("calibration_error", 1) <= 0.1)),
        )
    ]
