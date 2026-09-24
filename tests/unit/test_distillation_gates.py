"""Quality cannot be traded for speed; benchmark records retain authentication and grants."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from adaptive_llm.contracts import (
    BenchmarkMeasurement,
    BenchmarkReport,
    BenchmarkSpecification,
    DatasetSpecification,
    InferenceResponse,
    PairedComparison,
    SourceWindow,
    Usage,
    now,
    uid,
)
from adaptive_llm.distillation.benchmark import SQLiteBenchmarkStore, measure, passes
from adaptive_llm.evaluation.runner import Case, Outcome
from adaptive_llm.evaluation.service import EvaluationDeployment
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.providers import FakeProvider
from adaptive_llm.routing import FoundationRouter
from adaptive_llm.storage.sqlite import control_database


def report():
    measurement = BenchmarkMeasurement(
        requests=8,
        successes=8,
        p50_latency_ms=5,
        p95_latency_ms=10,
        requests_per_second=100,
        peak_rss_bytes=100_000,
        total_cost_micros=80,
        cost_per_success_micros=10,
        input_micros_per_1000_tokens=10,
        output_micros_per_1000_tokens=10,
    )
    return BenchmarkReport(
        specification=BenchmarkSpecification(
            candidate_version=uid(), evaluation_id=uid(), requests=8
        ),
        tenant_ids=["synthetic-a"],
        candidate_artifact_digest="synthetic-student",
        teacher_version="synthetic-teacher",
        teacher_artifact_digest="synthetic-teacher-digest",
        dataset_content_digest="synthetic-dataset",
        request_mix_digest="synthetic-mix",
        student=measurement,
        teacher=measurement.model_copy(update={"p95_latency_ms": 20}),
        quality_comparison=PairedComparison(sample_size=8, mean_delta=0, ci_lower=0, ci_upper=0),
        latency_reduction_fraction=0.5,
        cost_reduction_fraction=0,
        passed=True,
        known_limitations=["Synthetic measurement fixture."],
    )


def test_faster_student_never_compensates_for_inferior_or_unmeasured_quality():
    measured = report()
    assert passes(measured)
    for comparison in (
        PairedComparison(sample_size=8, mean_delta=-0.1, ci_lower=-0.15, ci_upper=-0.05),
        PairedComparison(sample_size=8, mean_delta=0, ci_lower=-0.02, ci_upper=0.02),
        PairedComparison(sample_size=1, mean_delta=0, ci_lower=0, ci_upper=0),
        PairedComparison(sample_size=0),
    ):
        assert not passes(measured.model_copy(update={"quality_comparison": comparison}))
    assert not passes(measured.model_copy(update={"latency_reduction_fraction": 0.19}))
    assert passes(
        measured.model_copy(
            update={"latency_reduction_fraction": 0, "cost_reduction_fraction": 0.3}
        )
    )
    assert not passes(
        measured.model_copy(
            update={"student": measured.student.model_copy(update={"successes": 0})}
        )
    )
    assert passes(
        measured.model_copy(
            update={
                "student": measured.student.model_copy(
                    update={"successes": 7, "cost_per_success_micros": 12}
                ),
                "teacher": measured.teacher.model_copy(
                    update={"successes": 7, "cost_per_success_micros": 12}
                ),
            }
        )
    )


@pytest.mark.parametrize("error", [False, True])
async def test_benchmark_success_counts_completion_independently_of_quality(
    settings, inference_request, error
):
    tokens = [f"synthetic{i}" for i in range(20)]
    target = " ".join(tokens)
    answer = " ".join([*tokens[:17], "replacement1", "replacement2", "replacement3"])
    cases = [
        Case(
            item_id=f"synthetic-{i}",
            tenant_id="synthetic-a",
            request=inference_request,
            target=target,
            expected_facts=("SYNTHETIC missing fact",),
        )
        for i in range(4)
    ]

    class CompletedRunner:
        async def run(self, case):
            failed = error and case.item_id == "synthetic-3"
            response = (
                None
                if failed
                else InferenceResponse(
                    interaction_id=uid(),
                    trace_id=uid(),
                    model_deployment_id="synthetic",
                    content=answer,
                    citations=[],
                    usage=Usage(
                        input_tokens=20,
                        output_tokens=20,
                        source="locally_estimated",
                        tokenizer="synthetic",
                    ),
                    estimated_cost_micros=10,
                    finish_reason="stop",
                )
            )
            return Outcome(response, None, 1, error=failed, cost_micros=10)

    deployment = EvaluationDeployment(
        FoundationRouter(settings.routing_path).deployment, FakeProvider()
    )
    measured, scores = await measure(CompletedRunner(), deployment, cases, 4)
    assert scores[:3] == pytest.approx([0.85] * 3)
    assert scores[3] == pytest.approx(0 if error else 0.85)
    assert measured.successes == (3 if error else 4)
    assert measured.total_cost_micros == 40
    assert measured.cost_per_success_micros == (14 if error else 10)


def test_benchmark_mac_identity_binding_immutability_and_grants(tmp_path, keyring, identity):
    database = control_database(tmp_path)
    store = SQLiteBenchmarkStore(database, keyring)
    operator = replace(identity, key_class="operator", dataset_tenants=frozenset({"synthetic-a"}))
    measured = report()
    try:
        saved = store.publish(measured, operator)
        assert saved.mac
        assert store.get(measured.specification.benchmark_id, operator) == saved
        assert (
            store.get(
                measured.specification.benchmark_id, replace(operator, dataset_tenants=frozenset())
            )
            is None
        )
        with pytest.raises(GatewayError, match="operator_required"):
            store.get(measured.specification.benchmark_id, identity)
        with pytest.raises(GatewayError, match="benchmark_exists"):
            store.publish(measured, operator)
        assert store.for_evaluation(
            measured.specification.candidate_version, measured.specification.evaluation_id, operator
        ) == [saved]
        other = uid()
        database.connection.execute("UPDATE benchmark_reports SET evaluation_id=?", (other,))
        assert store.for_evaluation(measured.specification.candidate_version, other, operator) == []
        changed = saved.model_copy(update={"cost_reduction_fraction": 1})
        database.connection.execute(
            "UPDATE benchmark_reports SET report=?", (changed.model_dump_json(),)
        )
        with pytest.raises(GatewayError, match="benchmark_integrity_failed"):
            store.get(measured.specification.benchmark_id, operator)
    finally:
        database.close()


def test_distillation_requires_source_and_teacher_and_bounds_generation():
    from datetime import timedelta

    spec = dict(
        dataset_id="synthetic-student",
        purpose="distillation",
        tenant_ids=["synthetic-a"],
        source_window=SourceWindow(start=now(), end=now() + timedelta(seconds=1)),
        eligibility_policy_version="synthetic",
    )
    with pytest.raises(ValidationError, match="distillation_source_and_teacher_required"):
        DatasetSpecification(**spec)
    spec.update(
        source_dataset_id="synthetic-source",
        source_dataset_version=uid(),
        teacher_deployment_id="fake-foundation-local-1",
    )
    assert DatasetSpecification(**spec).purpose == "distillation"
    with pytest.raises(ValidationError):
        DatasetSpecification(**spec, teacher_max_output_tokens=2049)
    with pytest.raises(ValidationError):
        BenchmarkSpecification(benchmark_id="..", candidate_version=uid(), evaluation_id=uid())
