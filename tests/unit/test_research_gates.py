"""Hardware measurements, not parameter estimates or configured prices, gate pruning."""

import pytest

from adaptive_llm.contracts import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSpecification,
    PairedComparison,
    uid,
)
from adaptive_llm.research.benchmark import passes


def measured():
    item = BenchmarkMeasurement(
        requests=8,
        successes=8,
        p50_latency_ms=10,
        p95_latency_ms=20,
        requests_per_second=50,
        peak_rss_bytes=1000,
        parameter_count=100,
        total_cost_micros=1,
        cost_per_success_micros=1,
        input_micros_per_1000_tokens=1,
        output_micros_per_1000_tokens=1,
    )
    return BenchmarkReport(
        specification=BenchmarkSpecification(
            candidate_version=uid(), evaluation_id=uid(), requests=8
        ),
        tenant_ids=["synthetic"],
        candidate_artifact_digest="synthetic-candidate",
        teacher_version="synthetic-base",
        teacher_artifact_digest="synthetic-base",
        dataset_content_digest="synthetic-data",
        request_mix_digest="synthetic-mix",
        student=item.model_copy(update={"parameter_count": 75}),
        teacher=item,
        quality_comparison=PairedComparison(sample_size=8, mean_delta=0, ci_lower=0, ci_upper=0),
        latency_reduction_fraction=0,
        peak_rss_reduction_fraction=0,
        cost_reduction_fraction=0.99,
        pruning_study_id=uid(),
        passed=False,
        known_limitations=["Synthetic numbers to test decision boundaries."],
    )


@pytest.mark.parametrize("field", ["latency_reduction_fraction", "peak_rss_reduction_fraction"])
def test_parameter_or_cost_reduction_alone_never_passes(field):
    report = measured()
    assert not passes(report)
    assert not passes(report.model_copy(update={field: 0.199}))
    assert passes(report.model_copy(update={field: 0.2}))
    for change in (
        {
            "quality_comparison": PairedComparison(
                sample_size=8, mean_delta=-0.03, ci_lower=-0.03, ci_upper=-0.03
            )
        },
        {"student": report.student.model_copy(update={"successes": 0})},
        {"student": report.student.model_copy(update={"parameter_count": None})},
        {"pruning_study_id": None},
    ):
        assert not passes(report.model_copy(update={field: 0.5, **change}))
