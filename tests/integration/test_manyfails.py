import hashlib
import json
from dataclasses import replace
from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import ROOT, Settings, create_app
from adaptive_llm.contracts import (
    DatasetManifest,
    DatasetSpecification,
    EvaluationInput,
    EvaluationReport,
    InferenceResponse,
    Interaction,
    ResponseFormat,
    RouteDecision,
    SourceWindow,
    TimeSplit,
    now,
    uid,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.evaluation.data import fixture_cases
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.evaluation.runner import Outcome
from adaptive_llm.evaluation.service import EvaluationDeployment
from adaptive_llm.evaluation.suites.scoring import assertions
from adaptive_llm.policy import ProcessingRedactor
from adaptive_llm.providers import FakeProvider, ProviderRequest
from adaptive_llm.providers.cassette import CassetteMissing, CassetteProvider
from adaptive_llm.providers.openrouter import OpenRouterProvider
from adaptive_llm.routing import FoundationRouter
from adaptive_llm.validation import LocalValidator

KEY = "SYNTHETIC-manyfails-key"
USER = {"Authorization": f"Bearer {KEY}"}
OPERATOR = {"Authorization": "Bearer synthetic-operator-key"}
MODEL = "anthropic/claude-haiku-4.5"
GOLDEN = fixture_cases(
    ROOT / "tests/fixtures/golden/manyfails-triage.jsonl", "manyfails", "research-sweep"
)
SAFETY = fixture_cases(
    ROOT / "tests/fixtures/safety/manyfails-triage.jsonl", "manyfails", "research-sweep"
)


def settings_for(tmp_path, provider=None):
    config = json.loads((ROOT / "configs/identity/manyfails.json").read_text())
    config["keys"]["manyfails-research-sweep"]["key_sha256"] = hashlib.sha256(
        KEY.encode()
    ).hexdigest()
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(config))
    return Settings(
        secret=b"SYNTHETIC-manyfails-test-secret-32-bytes",
        data_dir=tmp_path,
        identity_path=identity,
        policy_path=ROOT / "configs/policy/manyfails.json",
        routing_path=ROOT / "configs/routing/manyfails.json",
        tasks_path=ROOT / "configs/tasks/manyfails.json",
        provider=provider,
        outbox_dispatch_enabled=False,
    )


def canonical_case(case):
    request, _ = ProcessingRedactor().redact(case.request)
    return ProviderRequest(
        messages=tuple(request.messages),
        context=(),
        response_format=request.response_format,
        max_output_tokens=request.max_output_tokens,
        application_id=request.application_id,
        deadline_ms=request.routing.deadline_ms,
    )


class FixedVerdicts:
    def __init__(self, attack=False):
        self.targets = {
            canonical_case(c).messages[-1].content: c.target for c in [*GOLDEN, *SAFETY]
        }
        self.seen = []
        self.attack = attack

    async def generate(self, request):
        self.seen.append(request)
        content = self.targets[request.messages[-1].content]
        if self.attack:
            verdict = json.loads(content)
            verdict.update(is_failure=True, kind="product_shutdown")
            content = json.dumps(verdict)
        return replace(await FakeProvider().generate(request), content=content, citations=())


def evaluation_request():
    return EvaluationInput.model_validate_json(
        (ROOT / "configs/evaluation/manyfails-triage.json").read_text()
    )


def test_named_build_decontaminates_only_its_golden_set(tmp_path):
    provider = FixedVerdicts()
    # Contamination depends on the input, even when the model returns a different verdict.
    provider.targets[canonical_case(GOLDEN[0]).messages[-1].content] = SAFETY[0].target
    synthetic = fixture_cases(
        ROOT / "tests/fixtures/golden/synthetic.jsonl", "manyfails", "research-sweep"
    )[0]
    synthetic = replace(
        synthetic,
        request=synthetic.request.model_copy(
            update={"rag": SAFETY[0].request.rag, "response_format": ResponseFormat()}
        ),
    )
    provider.targets[synthetic.request.messages[-1].content] = synthetic.target
    app = create_app(settings_for(tmp_path, provider))
    start = now() - timedelta(seconds=1)
    with TestClient(app) as client:
        ids = []
        for case in (GOLDEN[0], synthetic):
            result = client.post(
                "/v1/inference", headers=USER, json=case.request.model_dump(mode="json")
            )
            assert result.status_code == 200
            ids.append(result.json()["interaction_id"])
        spec = DatasetSpecification(
            dataset_id="synthetic-decontamination",
            fixture_set="manyfails-triage",
            tenant_ids=["manyfails"],
            source_window=SourceWindow(start=start, end=now()),
            eligibility_policy_version="manyfails-1",
        )
        built = client.post(
            "/v1/datasets/builds", headers=OPERATOR, json=spec.model_dump(mode="json")
        )
        assert built.status_code == 200
        manifest = DatasetManifest.model_validate(built.json())
        assert manifest.quality_summary.exclusions["benchmark_contamination"] == 1
        shards = read_shards(manifest, tmp_path, app.state.persistence.cipher, app.state.keyring)
        rows = [json.loads(row) for values in shards.values() for row in values]
        assert [row["interaction_id"] for row in rows] == [ids[1]]
        assert manifest.specification.fixture_set == "manyfails-triage"


@pytest.mark.parametrize("attack", [False, True])
def test_five_suite_specialist_uses_named_structured_fixtures_and_dataset(tmp_path, attack):
    foundation = FixedVerdicts()
    specialist = FixedVerdicts()
    if attack:
        for case in SAFETY:
            target = json.loads(case.target)
            target.update(is_failure=True, kind="product_shutdown")
            specialist.targets[canonical_case(case).messages[-1].content] = json.dumps(target)
    settings = settings_for(tmp_path, foundation)
    deployment = FoundationRouter(settings.routing_path).deployment
    settings = replace(
        settings,
        evaluation_deployments={
            deployment.model_deployment_id: EvaluationDeployment(deployment, foundation),
            "synthetic-triage-specialist": EvaluationDeployment(
                deployment.model_copy(
                    update={"model_deployment_id": "synthetic-triage-specialist"}
                ),
                specialist,
            ),
        },
    )
    app = create_app(settings)
    start = now() - timedelta(seconds=1)
    with TestClient(app) as client:
        for index in range(9):
            body = SAFETY[0].request.model_dump(mode="json")
            body["request_id"] = uid()
            hit = json.loads(body["messages"][-1]["content"])
            hit["title"] = f"SYNTHETIC held-out unique headline {index}"
            body["messages"][-1]["content"] = json.dumps(hit)
            for provider in (foundation, specialist):
                provider.targets[body["messages"][-1]["content"]] = SAFETY[0].target
            result = client.post("/v1/inference", headers=USER, json=body)
            assert result.status_code == 200
            if index == 0:
                cutoff = now()
        dataset = DatasetSpecification(
            dataset_id="synthetic-triage-specialist",
            fixture_set="manyfails-triage",
            tenant_ids=["manyfails"],
            source_window=SourceWindow(start=start, end=now()),
            eligibility_policy_version="manyfails-1",
            near_duplicate_threshold=1,
            time_split=TimeSplit(
                train_end=cutoff, validation_end=cutoff + timedelta(microseconds=1)
            ),
        )
        built = client.post(
            "/v1/datasets/builds", headers=OPERATOR, json=dataset.model_dump(mode="json")
        )
        assert built.status_code == 200
        manifest = DatasetManifest.model_validate(built.json())
        spec = EvaluationInput(
            candidate_deployment_id=deployment.model_deployment_id,
            baseline_deployment_id=deployment.model_deployment_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.version,
            application_id="research-sweep",
            fixture_set="manyfails-triage",
            suites=["golden", "held_out", "safety", "retrieval", "performance"],
            minimum_sample_size=8,
            performance_requests=8,
        )
        baseline = client.post(
            "/v1/evaluations", headers=OPERATOR, json=spec.model_dump(mode="json")
        )
        assert baseline.status_code == 200
        assert baseline.json()["passed"]
        spec = spec.model_copy(
            update={
                "evaluation_id": uid(),
                "candidate_deployment_id": "synthetic-triage-specialist",
            }
        )
        result = client.post("/v1/evaluations", headers=OPERATOR, json=spec.model_dump(mode="json"))
        assert result.status_code == 200
        report = EvaluationReport.model_validate(result.json())
        assert report.coverage == {
            "golden": 30,
            "safety": 8,
            "held_out": 8,
            "retrieval": 8,
            "performance": 8,
        }
        suites = {suite.suite: suite for suite in report.suite_results}
        assert suites["golden"].metrics["rubric_score"] == 1
        assert suites["safety"].metrics["critical_failures"] == (8 if attack else 0)
        assert report.passed is (not attack)
        assert "required_suites" in {gate.gate for gate in report.gate_decisions}
        assert report.dataset_content_digest == manifest.content_digest
        from adaptive_llm.evaluation.data import fixture_digest

        assert report.suite_content_digest == fixture_digest(
            [
                ROOT / "tests/fixtures" / suite / "manyfails-triage.jsonl"
                for suite in ("golden", "safety")
            ]
        )
        assert (
            sum("held-out unique headline" in r.messages[-1].content for r in specialist.seen) == 24
        )


def test_hosted_startup_requires_key_and_health_is_truthful(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = settings_for(tmp_path)
    with pytest.raises(ValueError, match="^openrouter_api_key_required$"):
        with TestClient(create_app(settings)):
            pass
    with TestClient(create_app(replace(settings, inference_enabled=False))) as client:
        assert client.get("/healthz").json()["inference_enabled"] is False
        assert client.post("/v1/inference").status_code == 404


def test_manyfails_inference_redaction_schema_events_replay_correction_and_dataset(
    tmp_path, caplog
):
    case = SAFETY[0]
    target = case.target
    seen = []

    def transport(sent):
        body = json.loads(sent.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "```json\n" + target + "\n```"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            },
        )

    provider = OpenRouterProvider(MODEL, KEY, transport=httpx.MockTransport(transport))
    app = create_app(settings_for(tmp_path, provider))
    start = now() - timedelta(seconds=1)
    body = case.request.model_dump(mode="json")
    body["messages"][0]["content"] = (
        "SYNTHETIC system policy sk-synthetic123456789 contact synthetic@example.invalid"
    )
    body["metadata"] = {"sweep_query_id": "synthetic-sweep", "hit_url_sha256": "a" * 64}
    body["response_format"]["json_schema"]["schema"]["description"] = "SYNTHETIC-PRIVATE-SCHEMA"
    with TestClient(app) as client:
        first = client.post("/v1/inference", json=body, headers=USER)
        assert first.status_code == 200
        response = InferenceResponse.model_validate(first.json())
        assert json.loads(response.content) == json.loads(target)
        assert response.estimated_cost_micros == 300
        assert client.post("/v1/inference", json=body, headers=USER).json()["replayed"]
        assert len(seen) == 1
        assert seen[0]["messages"][0]["role"] == "system"
        assert "sk-synthetic" not in seen[0]["messages"][0]["content"]
        assert "synthetic@example.invalid" in seen[0]["messages"][0]["content"]
        conflict = {**body, "max_output_tokens": 401}
        assert client.post("/v1/inference", json=conflict, headers=USER).status_code == 409
        interaction = app.state.metadata.get("manyfails", Interaction, response.interaction_id)
        assert (
            interaction.task.label,
            interaction.task.classifier_version,
            interaction.task.risk_tier,
        ) == ("candidate_triage", "rules-1", "low")
        assert interaction.policy.content_logging_allowed and interaction.policy.training_allowed
        assert interaction.policy.persistence_redaction_counts["emails"] == 1
        blob = app.state.payloads.get("manyfails", interaction.input.messages_ref, now())
        stored = app.state.persistence.cipher.decrypt(
            blob, "manyfails", response.interaction_id, "messages"
        ).decode()
        assert json.loads(stored)[0]["role"] == "system"
        assert "synthetic@example.invalid" not in stored and "sk-synthetic" not in stored
        schema_blob = app.state.payloads.get(
            "manyfails", interaction.input.response_format_ref, now()
        )
        recorded_format = app.state.persistence.cipher.decrypt(
            schema_blob, "manyfails", response.interaction_id, "response_format"
        )
        assert json.loads(recorded_format) == body["response_format"]
        assert b"SYNTHETIC-PRIVATE-SCHEMA" not in schema_blob.ciphertext
        route = app.state.metadata.get("manyfails", RouteDecision, interaction.route_decision_id)
        assert route.response_schema_name == "triage" and len(route.response_schema_sha256) == 64
        while app.state.dispatcher.dispatch_once():
            pass
        telemetry = " ".join(e.model_dump_json() for e in app.state.events.events) + caplog.text
        assert "SYNTHETIC-PRIVATE-SCHEMA" not in telemetry
        assert "synthetic@example.invalid" not in telemetry
        assert target not in telemetry and KEY not in telemetry
        event_route = next(
            e.data for e in app.state.events.events if e.event_type == "route.decided.v1"
        )
        assert event_route == route
        correction = client.post(
            f"/v1/interactions/{response.interaction_id}/correction",
            headers=USER,
            json={"correction": target, "training_authorised": True},
        )
        assert correction.status_code == 200
        # Unique synthetic corrected example checks the system role all the way into a shard.
        spec = DatasetSpecification(
            dataset_id="synthetic-system",
            tenant_ids=["manyfails"],
            source_window=SourceWindow(start=start, end=now() + timedelta(seconds=1)),
            eligibility_policy_version="manyfails-1",
            minimum_examples=1,
        )
        built = client.post(
            "/v1/datasets/builds", headers=OPERATOR, json=spec.model_dump(mode="json")
        )
        assert built.status_code == 200
        manifest = DatasetManifest.model_validate(built.json())
        shards = read_shards(manifest, tmp_path, app.state.persistence.cipher, app.state.keyring)
        rows = [json.loads(row) for values in shards.values() for row in values]
        assert len(rows) == 1 and rows[0]["input"]["messages"][0]["role"] == "system"
    assert not provider._clients


@pytest.mark.parametrize("target", ["invalid", "{}", "[]", '{"is_failure":"yes"}'])
def test_invalid_correction_has_fixed_code_and_no_writes(tmp_path, target, caplog):
    app = create_app(settings_for(tmp_path, FixedVerdicts()))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", headers=USER, json=SAFETY[0].request.model_dump(mode="json")
        )
        iid = response.json()["interaction_id"]
        before = app.state.database.connection.total_changes
        result = client.post(
            f"/v1/interactions/{iid}/correction",
            headers=USER,
            json={"correction": target, "training_authorised": True},
        )
        assert result.status_code == 422
        assert result.json() == {"error": {"code": "invalid_correction_target"}}
        assert app.state.database.connection.total_changes == before
        assert target not in caplog.text


def test_recorded_format_deleted_and_missing_schema_fails_closed(tmp_path):
    app = create_app(settings_for(tmp_path, FixedVerdicts()))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", headers=USER, json=SAFETY[0].request.model_dump(mode="json")
        )
        iid = response.json()["interaction_id"]
        interaction = app.state.metadata.get("manyfails", Interaction, iid)
        ref = interaction.input.response_format_ref
        assert ref is not None
        with app.state.metadata.transaction():
            app.state.database.connection.execute(
                "DELETE FROM payloads WHERE reference = ?", (ref,)
            )
        result = client.post(
            f"/v1/interactions/{iid}/correction",
            headers=USER,
            json={"correction": SAFETY[0].target, "training_authorised": True},
        )
        assert (
            result.status_code == 503
            and result.json()["error"]["code"] == "correction_schema_unavailable"
        )
        assert client.delete(f"/v1/privacy/interactions/{iid}", headers=USER).status_code == 204
        assert (
            app.state.metadata.get("manyfails", Interaction, iid).input.response_format_ref is None
        )
        assert app.state.payloads.get("manyfails", ref, now()) is None


def test_schema_redaction_fails_closed_for_persistence_only(tmp_path):
    app = create_app(settings_for(tmp_path, FixedVerdicts()))
    body = SAFETY[0].request.model_dump(mode="json")
    body["response_format"]["json_schema"]["schema"]["description"] = "synthetic@example.invalid"
    with TestClient(app) as client:
        response = client.post("/v1/inference", headers=USER, json=body)
        assert response.status_code == 200
        interaction = app.state.metadata.get(
            "manyfails", Interaction, response.json()["interaction_id"]
        )
        assert interaction.error_code == "persistence_redaction_failed"
        assert (
            interaction.input.messages_ref is None and interaction.input.response_format_ref is None
        )


def test_correction_revalidated_after_redaction(tmp_path):
    class Redactor:
        version = "synthetic-1"

        def redact_text(self, value, policy):
            return ("invalid" if value == SAFETY[0].target else value), {}

    app = create_app(settings_for(tmp_path, FixedVerdicts()))
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference", headers=USER, json=SAFETY[0].request.model_dump(mode="json")
        )
        iid = response.json()["interaction_id"]
        app.state.persistence.redactor = Redactor()
        result = client.post(
            f"/v1/interactions/{iid}/correction",
            headers=USER,
            json={"correction": SAFETY[0].target, "training_authorised": True},
        )
        assert (
            result.status_code == 422
            and result.json()["error"]["code"] == "invalid_correction_target"
        )


def test_structured_fixture_baseline_lock_and_privacy_with_fixed_verdicts(tmp_path):
    provider = FixedVerdicts()
    app = create_app(settings_for(tmp_path, provider))
    spec = evaluation_request()
    with TestClient(app) as client:
        result = client.post("/v1/evaluations", headers=OPERATOR, json=spec.model_dump(mode="json"))
        assert result.status_code == 200
        report = EvaluationReport.model_validate(result.json())
        assert report.passed and report.paired_comparison.sample_size == 30
        assert report.suite_results[0].metrics["rubric_score"] == 1
        assert report.coverage == {"golden": 30, "safety": 8}
        assert all(
            r.application_id == "research-sweep"
            and r.messages[0].role == "system"
            and not r.context
            for r in provider.seen
        )
        for table in ("records", "payloads", "replays", "outbox"):
            # Tenant tables must remain empty during private evaluation.
            names = {
                r[0]
                for r in app.state.database.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if table in names:
                assert (
                    app.state.database.connection.execute(
                        f"SELECT count(*) FROM {table}"
                    ).fetchone()[0]
                    == 0
                )
        assert (
            app.state.evaluation_database.connection.execute(
                "SELECT count(*) FROM baselines"
            ).fetchone()[0]
            == 1
        )
        assert (
            client.post(
                "/v1/evaluations", headers=OPERATOR, json=spec.model_dump(mode="json")
            ).json()
            == result.json()
        )
        changed = spec.model_copy(update={"evaluation_id": uid()})
        assert (
            client.post(
                "/v1/evaluations", headers=OPERATOR, json=changed.model_dump(mode="json")
            ).status_code
            == 409
        )
        encoded = report.model_dump_json()
        assert all(
            c.request.messages[-1].content not in encoded and c.target not in encoded
            for c in GOLDEN
        )
        assert not all(
            g.passed
            for g in decisions(
                report.model_copy(
                    update={
                        "paired_comparison": report.paired_comparison.model_copy(
                            update={"sample_size": 29}
                        )
                    }
                )
            )
        )


def test_injected_page_verdict_blocks_lock_and_tenant_grants(tmp_path):
    app = create_app(settings_for(tmp_path, FixedVerdicts(attack=True)))
    with TestClient(app) as client:
        spec = evaluation_request()
        result = client.post("/v1/evaluations", headers=OPERATOR, json=spec.model_dump(mode="json"))
        assert result.status_code == 200
        report = EvaluationReport.model_validate(result.json())
        assert not report.passed
        safety = next(s for s in report.suite_results if s.suite == "safety")
        assert (
            safety.metrics["critical_failures"] == 8
            and safety.metrics["injection_success_rate"] == 1
        )
        assert (
            app.state.evaluation_database.connection.execute(
                "SELECT count(*) FROM baselines"
            ).fetchone()[0]
            == 0
        )
        denied = spec.model_copy(update={"evaluation_id": uid(), "fixture_tenant_id": "forbidden"})
        assert (
            client.post(
                "/v1/evaluations", headers=OPERATOR, json=denied.model_dump(mode="json")
            ).status_code
            == 403
        )
        denied_app = spec.model_copy(update={"evaluation_id": uid(), "application_id": "forbidden"})
        result = client.post(
            "/v1/evaluations", headers=OPERATOR, json=denied_app.model_dump(mode="json")
        )
        assert result.status_code == 403
        assert result.json() == {"error": {"code": "application_forbidden"}}


@pytest.mark.parametrize("case", [*GOLDEN, *SAFETY], ids=lambda c: c.item_id)
async def test_recorded_manyfails_case(case):
    provider = CassetteProvider(MODEL)
    request = canonical_case(case)
    try:
        provider.require(request)
    except CassetteMissing as error:
        pytest.skip(str(error))
    result = await provider.generate(request)
    assert LocalValidator().validate(request, result).passed
    if case.critical:
        response = InferenceResponse(
            interaction_id=uid(),
            trace_id=uid(),
            model_deployment_id="cassette",
            content=result.content,
            citations=list(result.citations),
            usage=result.usage,
            finish_reason=result.finish_reason,
            estimated_cost_micros=0,
        )
        assert all(assertions(case, Outcome(response, None, 0)).values())


def test_recorded_manyfails_baseline_lock(tmp_path):
    provider = CassetteProvider(MODEL)
    for case in [*GOLDEN, *SAFETY]:
        try:
            provider.require(canonical_case(case))
        except CassetteMissing as error:
            pytest.skip(str(error))
    app = create_app(settings_for(tmp_path, provider))
    with TestClient(app) as client:
        result = client.post(
            "/v1/evaluations", headers=OPERATOR, json=evaluation_request().model_dump(mode="json")
        )
        assert result.status_code == 200
        report = EvaluationReport.model_validate(result.json())
        assert report.passed and report.paired_comparison.sample_size == 30


@pytest.mark.parametrize(
    "status,http_status,code",
    [
        (429, 503, "provider_rate_limited"),
        (503, 503, "provider_unavailable"),
        (504, 504, "provider_timeout"),
        (200, 502, "provider_invalid_response"),
    ],
)
def test_provider_failure_codes_survive_pipeline_without_content(
    tmp_path, caplog, status, http_status, code
):
    private = "SYNTHETIC-private-provider-body"
    provider = OpenRouterProvider(
        MODEL,
        KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text=private, headers={"X-Private": KEY})
        ),
    )
    app = create_app(settings_for(tmp_path, provider))
    with TestClient(app) as client:
        result = client.post(
            "/v1/inference", json=SAFETY[0].request.model_dump(mode="json"), headers=USER
        )
        assert result.status_code == http_status
        assert result.json() == {"error": {"code": code}}
        while app.state.dispatcher.dispatch_once():
            pass
        telemetry = caplog.text + " ".join(e.model_dump_json() for e in app.state.events.events)
        assert private not in telemetry and KEY not in telemetry
        failed = next(
            e.data for e in app.state.events.events if e.event_type == "generation.failed.v1"
        )
        assert failed.error_code == code


def test_invalid_schema_http_error_has_no_schema_or_prompt(tmp_path, caplog):
    app = create_app(settings_for(tmp_path, FixedVerdicts()))
    body = SAFETY[0].request.model_dump(mode="json")
    body["response_format"]["json_schema"]["schema"] = {"type": "SYNTHETIC-private-invalid-schema"}
    with TestClient(app) as client:
        result = client.post("/v1/inference", json=body, headers=USER)
        assert result.status_code == 422 and result.json() == {"error": {"code": "invalid_request"}}
        assert "SYNTHETIC-private-invalid-schema" not in result.text + caplog.text
        assert not app.state.events.events
