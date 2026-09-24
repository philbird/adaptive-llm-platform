from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from adaptive_llm.contracts import (
    EvaluationCompleted,
    EvaluationInput,
    EvaluationReport,
    Event,
    PairedComparison,
    uid,
)
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.service import EvaluationDeployment
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult

if TYPE_CHECKING:
    from conftest import EvaluationSeed

OPERATOR = {"Authorization": "Bearer synthetic-operator-key", "X-Subject": "synthetic-evaluator"}


def evaluate(seed: "EvaluationSeed", request: EvaluationInput | None = None) -> EvaluationReport:
    result = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=(request or seed.request).model_dump(mode="json")
    )
    assert result.status_code == 200, result.json()
    return EvaluationReport.model_validate(result.json())


def test_all_suites_baseline_lock_mac_events_idempotency_and_replacement(
    evaluation_seed: "EvaluationSeed",
) -> None:
    seed = evaluation_seed
    report = evaluate(seed)
    assert report.passed, report.gate_decisions
    assert report.paired_comparison == PairedComparison(
        mean_delta=0, ci_lower=0, ci_upper=0, sample_size=8
    )
    assert all(value.sample_size == 8 for value in report.segment_comparisons.values())
    suites = {s.suite: s for s in report.suite_results}
    assert suites["golden"].items == 22
    assert suites["safety"].items == 20
    assert suites["golden"].metrics["rubric_score"] == 5
    assert suites["golden"].metrics["judge_disagreements"] == 0
    assert suites["held_out"].metrics == {
        "token_f1": 1,
        "citation_precision": 1,
        "citation_recall": 1,
    }
    assert suites["retrieval"].metrics == {
        "recall_at_k": 1,
        "mrr": 1,
        "context_precision": 1,
        "acl_incidents": 0,
    }
    assert suites["performance"].first_token_latency_ms is None
    assert isinstance(suites["performance"].cost_per_success_micros, int)
    assert evaluate(seed) == report
    assert seed.client.get(
        f"/v1/evaluations/{seed.request.evaluation_id}", headers=OPERATOR
    ).json() == report.model_dump(mode="json")
    directory = seed.directory / "evaluations" / seed.request.evaluation_id
    encoded = (directory / "report.json").read_text()
    from adaptive_llm.signing import FIELDS, verify_record

    verify_record(
        report,
        seed.app.state.keyring.verifier,
        "evaluation-report",
        report.model_dump_json(exclude=FIELDS),
    )
    assert report.signature in encoded
    assert (directory / "report.mac").read_text() == ""
    replacement = seed.request.model_copy(update={"evaluation_id": uid()})
    assert (
        seed.client.post(
            "/v1/evaluations", headers=OPERATOR, json=replacement.model_dump(mode="json")
        ).status_code
        == 409
    )
    replacement = replacement.model_copy(
        update={"replace": True, "operator_note": "SYNTHETIC replacement note"}
    )
    second = evaluate(seed, replacement)
    assert second.passed
    connection = seed.app.state.evaluation_database.connection
    row = connection.execute("SELECT * FROM baselines").fetchone()
    assert row["evaluation_id"] == replacement.evaluation_id
    assert connection.execute("SELECT count(*) FROM evaluation_reports").fetchone()[0] == 2
    assert connection.execute(
        "SELECT operator_note_hash FROM evaluation_reports WHERE evaluation_id=?",
        (replacement.evaluation_id,),
    ).fetchone()[0] == seed.app.state.keyring.fingerprint(replacement.operator_note)
    while seed.app.state.evaluation_dispatcher.dispatch_once():
        pass
    events = [
        e
        for e in seed.app.state.evaluation_events.events
        if isinstance(e.data, EvaluationCompleted)
    ]
    assert len(events) == 4
    assert {e.tenant_id for e in events} == {"synthetic-a", "synthetic-b"}
    assert all(e.data.report_ref.startswith("evaluations/") for e in events)
    assert len({e.event_id for e in events}) == 4
    assert len({e.trace_id for e in events}) == 2
    assert seed.app.state.events.events == []
    assert (
        seed.app.state.database.connection.execute(
            "SELECT count(*) FROM outbox WHERE event_type='evaluation.completed.v1'"
        ).fetchone()[0]
        == 0
    )
    print(
        f"baseline lock: dataset={seed.manifest.version} "
        f"evaluation={report.specification.evaluation_id} n=8 CI=[0,0] passed={report.passed}"
    )


class DropCitations:
    async def generate(self, request: ProviderRequest) -> ProviderResult:
        result = await FakeProvider().generate(request)
        text = result.content
        for citation in result.citations:
            text = text.replace(f"[{citation.document_id}/{citation.chunk_id}]", "")
        return replace(result, content=text, citations=())


def test_identical_candidate_passes_and_dropped_citations_fail_critical_segments(
    evaluation_seed: "EvaluationSeed",
) -> None:
    seed = evaluation_seed
    baseline = evaluate(seed)
    evaluator = seed.app.state.evaluations
    foundation = evaluator.deployments[seed.request.candidate_deployment_id]
    for name, provider, passed in [
        ("synthetic-identical", FakeProvider(), True),
        ("synthetic-worse", DropCitations(), False),
    ]:
        evaluator.deployments[name] = EvaluationDeployment(
            foundation.manifest.model_copy(
                update={"model_deployment_id": name, "model_version": name}
            ),
            provider,
        )
        request = seed.request.model_copy(
            update={"candidate_deployment_id": name, "evaluation_id": uid()}
        )
        report = evaluate(seed, request)
        assert report.passed is passed
        assert report.baseline_report_id == baseline.specification.evaluation_id
        if not passed:
            gates = {g.gate: g.passed for g in report.gate_decisions}
            assert not gates["segment.citation"] and not gates["segment.safety"]
            assert not gates["safety_assertions"] and not gates["safety_privacy_isolation"]
            assert report.paired_comparison.mean_delta == -1


def test_missing_small_sample_margin_and_coverage_fail_closed(
    evaluation_seed: "EvaluationSeed",
) -> None:
    seed = evaluation_seed
    report = evaluate(seed)
    for minimum in [None, 9]:
        changed = report.model_copy(
            update={
                "specification": report.specification.model_copy(
                    update={"minimum_sample_size": minimum}
                )
            }
        )
        gates = {g.gate: g.passed for g in decisions(changed)}
        assert not gates["sample_size"] and not gates["non_inferiority"]
    for lower in [-0.03, -0.02]:
        changed = report.model_copy(
            update={
                "paired_comparison": PairedComparison(
                    mean_delta=0, ci_lower=lower, ci_upper=0.01, sample_size=8
                )
            }
        )
        assert not next(g.passed for g in decisions(changed) if g.gate == "non_inferiority")
    changed = report.model_copy(update={"segment_comparisons": {}})
    assert not next(g.passed for g in decisions(changed) if g.gate == "segment.citation")
    changed = report.model_copy(update={"suite_results": report.suite_results[:-1]})
    assert not next(g.passed for g in decisions(changed) if g.gate == "required_suites")


@pytest.mark.parametrize("failure", ["rename", "outbox"])
def test_report_publication_is_atomic(
    evaluation_seed: "EvaluationSeed", monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    seed = evaluation_seed
    if failure == "rename":

        def fail(*args: object) -> None:
            raise OSError("synthetic failure")

        monkeypatch.setattr(Path, "rename", fail)
    else:
        original = seed.app.state.evaluation_outbox.enqueue

        def fail_events(events: object, limit: int) -> int:
            if any(e.event_type == "evaluation.completed.v1" for e in events):
                raise RuntimeError("synthetic failure")
            return original(events, limit)

        monkeypatch.setattr(seed.app.state.evaluation_outbox, "enqueue", fail_events)
    result = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=seed.request.model_dump(mode="json")
    )
    assert result.status_code == 503
    db = seed.app.state.evaluation_database.connection
    assert db.execute("SELECT count(*) FROM evaluation_reports").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM baselines").fetchone()[0] == 0
    assert (
        db.execute(
            "SELECT count(*) FROM outbox WHERE event_type='evaluation.completed.v1'"
        ).fetchone()[0]
        == 0
    )
    assert list((seed.directory / "evaluations").iterdir()) == []


def test_cli_and_api_share_report(
    evaluation_seed: "EvaluationSeed", capsys: pytest.CaptureFixture[str]
) -> None:
    from adaptive_llm.evaluation.__main__ import main

    seed = evaluation_seed
    path = seed.directory / "evaluation.json"
    path.write_text(seed.request.model_dump_json())
    main(["--spec", str(path), "--data-dir", str(seed.directory)])
    report = EvaluationReport.model_validate_json(capsys.readouterr().out)
    assert report == evaluate(seed)
    assert report.passed
    main_args = [
        "--spec",
        str(path),
        "--data-dir",
        str(seed.directory),
        "--operator-key",
        "synthetic-key-a",
    ]
    with pytest.raises(SystemExit) as error:
        main(main_args)
    assert error.value.code == 1
    assert capsys.readouterr().err == "evaluation_failed\n"


@pytest.mark.parametrize(
    "update",
    [{"minimum_sample_size": None}, {"minimum_sample_size": 9}, {"baseline_deployment_id": None}],
)
def test_missing_baseline_or_samples_persists_failure_without_lock(
    evaluation_seed: "EvaluationSeed", update: dict[str, object]
) -> None:
    seed = evaluation_seed
    report = evaluate(seed, seed.request.model_copy(update=update))
    assert not report.passed
    assert (
        seed.app.state.evaluation_database.connection.execute(
            "SELECT count(*) FROM baselines"
        ).fetchone()[0]
        == 0
    )
    assert (
        seed.app.state.evaluation_database.connection.execute(
            "SELECT count(*) FROM evaluation_reports"
        ).fetchone()[0]
        == 1
    )
    if update.get("baseline_deployment_id", "present") is None:
        rows = seed.app.state.evaluation_database.connection.execute("SELECT envelope FROM outbox")
        events = [Event.model_validate_json(row[0]) for row in rows]
        assert len(events) == 2
        assert all(
            isinstance(e.data, EvaluationCompleted) and e.data.baseline_version is None
            for e in events
        )


def test_baseline_manifest_pin_and_failed_replacement_preserve_previous_lock(
    evaluation_seed: "EvaluationSeed",
) -> None:
    seed = evaluation_seed
    baseline = evaluate(seed)
    failed = evaluate(
        seed,
        seed.request.model_copy(
            update={
                "evaluation_id": uid(),
                "replace": True,
                "operator_note": "SYNTHETIC test failed replacement",
                "minimum_sample_size": 9,
            }
        ),
    )
    assert not failed.passed
    assert (
        seed.app.state.evaluation_database.connection.execute(
            "SELECT evaluation_id FROM baselines"
        ).fetchone()[0]
        == baseline.specification.evaluation_id
    )
    evaluator = seed.app.state.evaluations
    foundation = evaluator.deployments[seed.request.candidate_deployment_id]
    evaluator.deployments["synthetic-candidate"] = EvaluationDeployment(
        foundation.manifest.model_copy(update={"model_deployment_id": "synthetic-candidate"}),
        FakeProvider(),
    )
    evaluator.deployments[seed.request.candidate_deployment_id] = EvaluationDeployment(
        foundation.manifest.model_copy(update={"model_version": "changed"}), FakeProvider()
    )
    request = seed.request.model_copy(
        update={"evaluation_id": uid(), "candidate_deployment_id": "synthetic-candidate"}
    )
    result = seed.client.post(
        "/v1/evaluations", headers=OPERATOR, json=request.model_dump(mode="json")
    )
    assert result.status_code == 409
    assert result.json() == {"error": {"code": "baseline_version_mismatch"}}


def test_evaluations_run_off_event_loop_and_concurrent_locks_have_one_winner(
    evaluation_seed: "EvaluationSeed",
) -> None:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    seed = evaluation_seed
    entered = Event()
    release = Event()
    mutex = Lock()
    calls = 0

    class BlockingFake:
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            nonlocal calls
            with mutex:
                calls += 1
                if calls >= 2:
                    entered.set()
            assert await asyncio.to_thread(release.wait, 5)
            return await FakeProvider().generate(request)

    evaluator = seed.app.state.evaluations
    original = evaluator.deployments[seed.request.candidate_deployment_id]
    evaluator.deployments[seed.request.candidate_deployment_id] = EvaluationDeployment(
        original.manifest, BlockingFake()
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(
            seed.client.post,
            "/v1/evaluations",
            headers=OPERATOR,
            json=seed.request.model_dump(mode="json"),
        )
        second = executor.submit(
            seed.client.post,
            "/v1/evaluations",
            headers=OPERATOR,
            json=seed.request.model_copy(update={"evaluation_id": uid()}).model_dump(mode="json"),
        )
        try:
            assert entered.wait(3)
            serving = executor.submit(
                seed.client.post,
                "/v1/inference",
                headers={"Authorization": "Bearer synthetic-key-a"},
                json={
                    "request_id": uid(),
                    "application_id": "support-assistant",
                    "messages": [{"role": "user", "content": "SYNTHETIC concurrent request"}],
                },
            )
            assert serving.result(timeout=1).status_code == 200
            assert not first.done() and not second.done()
        finally:
            release.set()
        responses = [first.result(timeout=10), second.result(timeout=10)]
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert (
        seed.app.state.evaluation_database.connection.execute(
            "SELECT count(*) FROM baselines"
        ).fetchone()[0]
        == 1
    )
    assert len(list((seed.directory / "evaluations").iterdir())) == 1


def test_candidate_and_baseline_share_one_event_loop(evaluation_seed: "EvaluationSeed") -> None:
    import asyncio

    loops = set()
    calls = 0

    class LoopBoundFake:
        async def generate(self, request: ProviderRequest) -> ProviderResult:
            nonlocal calls
            loops.add(asyncio.get_running_loop())
            calls += 1
            return await FakeProvider().generate(request)

    seed = evaluation_seed
    evaluator = seed.app.state.evaluations
    original = evaluator.deployments[seed.request.candidate_deployment_id]
    evaluator.deployments[seed.request.candidate_deployment_id] = EvaluationDeployment(
        original.manifest, LoopBoundFake()
    )
    assert evaluate(seed).passed
    assert calls == 2 * (22 + 8 + 20 + 8 + seed.request.performance_requests)
    assert len(loops) == 1
    assert all(loop.is_closed() for loop in loops)


@pytest.mark.parametrize("field", ["judge_version", "rubric_version"])
def test_unregistered_versions_fail_at_runtime(
    evaluation_seed: "EvaluationSeed", field: str
) -> None:
    seed = evaluation_seed
    body = seed.request.model_dump(mode="json") | {field: "synthetic-unregistered-2"}
    # The version is a valid extensible identifier, rejected by runtime registration.
    EvaluationInput.model_validate(body)
    response = seed.client.post("/v1/evaluations", headers=OPERATOR, json=body)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": f"{field}_mismatch"}}
    assert not list((seed.directory / "evaluations").glob("*/report.json"))


@pytest.mark.parametrize("enabled", [True, False])
def test_extensible_registered_judge_and_nullable_rubric(
    evaluation_seed: "EvaluationSeed",
    enabled: bool,
) -> None:
    from adaptive_llm.evaluation.judge import DeterministicJudge
    from adaptive_llm.evaluation.suites.golden import GoldenSuite

    class NextJudge(DeterministicJudge):
        version = "synthetic-next-judge-2"
        rubric_version = "synthetic-next-rubric-2"

    seed = evaluation_seed
    evaluator = seed.app.state.evaluations
    evaluator.judge = NextJudge()
    evaluator.suites["golden"] = GoldenSuite(evaluator.judge)
    request = seed.request.model_copy(
        update={
            "judge_version": evaluator.judge.version if enabled else None,
            "rubric_version": evaluator.judge.rubric_version if enabled else None,
        }
    )
    report = evaluate(seed, request)
    assert report.passed
    golden = next(s for s in report.suite_results if s.suite == "golden")
    assert ("rubric_score" in golden.metrics) is enabled
