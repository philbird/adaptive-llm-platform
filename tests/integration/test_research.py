"""Synthetic CPU studies, physical compaction and the ordinary candidate lifecycle."""

import asyncio
import json
from dataclasses import replace
from time import perf_counter

import pytest
from fastapi.testclient import TestClient
from test_distillation import smoke_suites
from test_training import OPERATOR, approve, evaluate, spec_for, train

from adaptive_llm.app import create_app
from adaptive_llm.contracts import (
    AdapterConfig,
    BenchmarkSpecification,
    DatasetApproval,
    DatasetManifest,
    Message,
    ModelManifest,
    PruningPlan,
    ResponseFormat,
    TrainingJob,
    uid,
)
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.providers import ProviderRequest
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.research.models import (
    ActivationSpecification,
    BaseSpecification,
    PruneSpecification,
)
from adaptive_llm.training.lora import LoraTrainer, generate_tiny_base, libraries

pytestmark = [
    pytest.mark.accelerator_free,
    pytest.mark.parametrize("evaluation_seed", ["no_context"], indirect=True),
]


@pytest.fixture
def research(evaluation_seed):
    try:
        libraries()
    except GatewayError:
        pytest.skip("optional training group absent")
    seed = evaluation_seed
    approve(seed)
    original = seed.app.state.settings
    routing = json.loads(original.routing_path.read_text())
    routing["pruning_research_enabled"] = True
    route_path = seed.directory / "research-routing.json"
    route_path.write_text(json.dumps(routing))
    auth = json.loads(original.identity_path.read_text())
    auth["keys"]["synthetic-research-key"] = {
        **auth["keys"]["synthetic-operator-key"],
        "capabilities": ["research"],
    }
    auth_path = seed.directory / "research-identity.json"
    auth_path.write_text(json.dumps(auth))
    app = create_app(
        replace(
            original,
            pruning_research_enabled=True,
            routing_path=route_path,
            identity_path=auth_path,
            policy=seed.app.state.training.policy,
        )
    )
    seed.app.state.training.stop()
    with TestClient(app) as client:
        seed.app, seed.client = app, client
        generate_tiny_base(seed.directory / "base-models/tiny/seed-17")
        service = app.state.training
        service.trainer = LoraTrainer(seed.directory, service.cipher, service.keyring)
        training = spec_for(seed).model_copy(
            update={
                "base_model_id": "tiny",
                "base_model_revision": "seed-17",
                "base_model_licence": "CC0-1.0",
                "tokenizer_id": "tiny-byte-v1",
                "chat_template_version": "tiny-chat-v1",
                "steps": 500,
                "checkpoint_every": 100,
                "max_sequence_length": 128,
                "adapter_config": AdapterConfig(
                    target_modules=["q_proj", "v_proj", "lm_head"], learning_rate=0.005
                ),
            }
        )
        teacher = train(seed, training)
        assert teacher.state == "succeeded", teacher.failure_code
        smoke_suites(seed)
        evaluate(seed)
        baseline = evaluate(seed, teacher.model_version)
        assert baseline.passed, baseline.model_dump(include={"gate_decisions", "suite_results"})
        spec = ActivationSpecification(
            base=BaseSpecification(adapter_version=teacher.model_version),
            calibration_dataset_id=seed.manifest.dataset_id,
            calibration_dataset_version=seed.manifest.version,
            evaluation_dataset_id=seed.manifest.dataset_id,
            evaluation_dataset_version=seed.manifest.version,
            baseline_evaluation_id=baseline.specification.evaluation_id,
            calibration_split="train",
        )
        yield seed, spec, training


RESEARCH = {"Authorization": "Bearer synthetic-research-key", "X-Subject": "synthetic-operator"}


def submit(seed, spec):
    response = seed.client.post(
        "/v1/research/jobs", headers=RESEARCH, json=spec.model_dump(mode="json")
    )
    assert response.status_code == 200, response.json()
    return response.json()


def research_identity(seed):
    return seed.app.state.authenticator.authenticate(
        RESEARCH["Authorization"], "synthetic-operator"
    )


def counts(seed):
    db = seed.app.state.metadata.database
    with db.lock:
        return {
            name: db.connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for (name,) in db.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }


def test_study_aggregates_encryption_authentication_isolation_and_rankings(research):
    seed, spec, _ = research
    before = counts(seed)
    summary = submit(seed, spec)
    assert counts(seed) == before
    assert summary["sample_size"] == 1
    assert summary["shapes"] == {
        "layers": [2, 1, 4],
        "attention_heads": [2, 4, 4],
        "mlp_channels": [2, 64, 4],
    }
    url = f"/v1/research/studies/{spec.study_id}"
    assert seed.client.get(url, headers=OPERATOR).status_code == 403
    assert seed.client.get(url + "/aggregates", headers=OPERATOR).status_code == 403
    raw = seed.client.get(url + "/aggregates", headers=RESEARCH).content
    tensors = libraries().tensors.load(raw)
    assert set(tensors) == {"layers", "attention_heads", "mlp_channels"}
    for key, tensor in tensors.items():
        assert list(tensor.shape) == summary["shapes"][key]
        assert tensor.isfinite().all()
        assert (tensor[..., 3] >= 0).all()
    assert any((t[..., 3] > 0).any() for t in tensors.values())
    path = seed.directory / "research/studies" / spec.study_id
    assert (path / "aggregates.safetensors.enc").read_bytes() != raw
    for p in path.iterdir():
        assert b"SYNTHETIC ANSWER" not in p.read_bytes()
    assert submit(seed, spec) == summary
    original = (path / "summary.json").read_bytes()
    changed = json.loads(original)
    changed["summary"]["rankings"]["layers"].reverse()
    (path / "summary.json").write_text(json.dumps(changed))
    assert (
        seed.client.get(url, headers=RESEARCH).json()["error"]["code"] == "study_integrity_failed"
    )
    (path / "summary.json").write_bytes(original)
    other = spec.model_copy(update={"study_id": uid()})
    submit(seed, other)
    other_url = f"/v1/research/studies/{other.study_id}/aggregates"
    assert seed.client.get(other_url, headers=RESEARCH).content == raw
    # Even a newly signed envelope cannot move ciphertext across the study AAD boundary.
    other_path = path.parent / other.study_id
    envelope = json.loads((other_path / "summary.json").read_bytes())
    signed = json.loads(original)
    for name in ("nonce", "key_version", "digest"):
        envelope[name] = signed[name]
    envelope.pop("mac")
    envelope["mac"] = seed.app.state.research.store._mac(envelope)
    (other_path / "summary.json").write_text(json.dumps(envelope))
    (other_path / "aggregates.safetensors.enc").write_bytes(
        (path / "aggregates.safetensors.enc").read_bytes()
    )
    assert (
        seed.client.get(other_url, headers=RESEARCH).json()["error"]["code"]
        == "study_integrity_failed"
    )
    (path / "aggregates.safetensors.enc").write_bytes(raw)
    result = seed.client.get(url + "/aggregates", headers=RESEARCH)
    assert result.json()["error"]["code"] == "study_integrity_failed"


@pytest.mark.parametrize(
    "failure",
    [
        "capability",
        "licence",
        "baseline",
        "failed_baseline",
        "approval",
        "evaluation_approval",
        "manifest",
        "policy",
    ],
)
def test_research_preconditions_refuse_before_artifacts(research, failure):
    seed, spec, _ = research
    headers = RESEARCH
    expected = "research_baseline_required"
    if failure == "capability":
        headers, expected = OPERATOR, "research_capability_required"
    elif failure == "licence":
        path = seed.directory / "licences.json"
        path.write_text('{"allowed": []}')
        seed.app.state.research.licences_path = path
        expected = "research_licence_denied"
    elif failure == "baseline":
        spec = spec.model_copy(update={"baseline_evaluation_id": uid()})
    elif failure == "failed_baseline":
        # Real raw random base is evaluated through the same five suites and fails them.
        response = seed.client.post(
            "/v1/research/baselines",
            headers=RESEARCH,
            json={
                "base": BaseSpecification().model_dump(mode="json"),
                "evaluation": seed.request.model_copy(update={"evaluation_id": uid()}).model_dump(
                    mode="json"
                ),
            },
        )
        assert response.status_code == 200, response.json()
        assert not response.json()["passed"]
        spec = spec.model_copy(
            update={
                "base": BaseSpecification(),
                "baseline_evaluation_id": response.json()["specification"]["evaluation_id"],
            }
        )
    elif failure == "approval":
        manifest = seed.app.state.metadata.get_manifest(
            seed.manifest.dataset_id, seed.manifest.version
        )
        # A pending approval is a valid immutable original manifest, not a forged approval.
        manifest = manifest.model_copy(update={"approval": DatasetApproval()})
        with seed.app.state.metadata.database.transaction():
            seed.app.state.metadata.approve_manifest(manifest)
        expected = "dataset_approval_required"
    elif failure == "evaluation_approval":
        response = seed.client.post(
            "/v1/datasets/builds",
            headers=OPERATOR,
            json=seed.manifest.specification.model_copy(
                update={"dataset_id": "synthetic-unapproved-evaluation"}
            ).model_dump(mode="json"),
        )
        assert response.status_code == 200, response.json()
        spec = spec.model_copy(
            update={
                "evaluation_dataset_id": response.json()["dataset_id"],
                "evaluation_dataset_version": response.json()["version"],
            }
        )
        expected = "dataset_approval_required"
    elif failure == "manifest":
        path = seed.directory / "datasets" / seed.manifest.dataset_id / seed.manifest.version
        p = next(path.glob("*.enc"))
        p.write_bytes(p.read_bytes() + b"synthetic-tamper")
        expected = "invalid_dataset_artifact"
    elif failure == "policy":
        seed.app.state.training.policy.training["synthetic-a"] = False
        expected = "training_policy_denied"
    response = seed.client.post(
        "/v1/research/jobs", headers=headers, json=spec.model_dump(mode="json")
    )
    assert response.json()["error"]["code"] == expected
    assert not (seed.directory / "research").exists()


@pytest.mark.smoke
def test_study_prune_finetune_evaluate_benchmark_smoke(research, monkeypatch):
    seed, spec, training = research
    start = perf_counter()
    submit(seed, spec)
    prune = PruneSpecification(
        study_id=spec.study_id,
        plan=PruningPlan(structures=["attention_heads", "mlp_channels"]),
        training=training.model_copy(update={"job_id": uid(), "registry_id": "synthetic-pruned"}),
    )
    before = counts(seed)
    model = ModelManifest.model_validate(submit(seed, prune))
    assert counts(seed) == before
    assert model.state == "candidate" and model.adapter_architecture == "pruned-full-v1"
    assert model.pruning.parameter_count_after < model.pruning.parameter_count_before
    assert len(model.pruning.removed_indices["attention_heads.0"]) == 1
    files = seed.directory / model.storage_location
    config = json.loads((files / "config.json").read_text())
    assert config["num_attention_heads"] == 3
    assert config["num_key_value_heads"] == 3
    assert config["intermediate_size"] == 48
    provider = SpecialistProvider(model, files, seed.app.state.keyring, seed.directory)
    result = asyncio.run(
        provider.generate(
            ProviderRequest(
                messages=(Message(role="user", content="SYNTHETIC test"),),
                context=(),
                max_output_tokens=64,
                response_format=ResponseFormat(),
            )
        )
    )
    assert result.usage.output_tokens > 0
    report = evaluate(seed, model.version)
    assert {s.suite for s in report.suite_results} == {
        "golden",
        "held_out",
        "safety",
        "retrieval",
        "performance",
    }
    benchmark_spec = BenchmarkSpecification(
        candidate_version=model.version,
        evaluation_id=report.specification.evaluation_id,
        requests=8,
    )

    def forbidden_load(*args, **kwargs):
        pytest.fail("benchmark pre-checks and promotion must not allocate models")

    # Actual measurement loads models only in fresh child processes.
    monkeypatch.setattr("adaptive_llm.research.service.BaseGenerator", forbidden_load)
    monkeypatch.setattr("adaptive_llm.research.service.LoraGenerator", forbidden_load)
    monkeypatch.setattr(seed.app.state.research.evaluator, "_deployment", forbidden_load)
    seed.app.state.research.identify_source(BaseSpecification(), research_identity(seed))
    response = seed.client.post(
        "/v1/benchmarks/jobs", headers=RESEARCH, json=benchmark_spec.model_dump(mode="json")
    )
    assert response.status_code == 200, response.json()
    benchmark = response.json()
    assert benchmark["student"]["parameter_count"] == model.pruning.parameter_count_after
    assert benchmark["teacher"]["parameter_count"] == model.pruning.parameter_count_before
    assert benchmark["teacher_artifact_digest"] == model.pruning.adapter_digest
    promotion = seed.client.post(
        f"/v1/models/{model.version}/promotion-requests",
        headers=RESEARCH,
        json={
            "model_version": model.version,
            "target_state": "approved",
            "evaluation_id": report.specification.evaluation_id,
            "reason": "SYNTHETIC research",
        },
    )
    assert (promotion.status_code == 200) == (report.passed and benchmark["passed"])
    summary = seed.client.get(f"/v1/research/studies/{spec.study_id}", headers=RESEARCH).json()
    assert summary["candidates"][model.version] == model.pruning.parameter_count_after
    assert summary["benchmark_ids"] == [benchmark_spec.benchmark_id]
    assert summary["evaluation_ids"] == [report.specification.evaluation_id]
    markdown = (seed.directory / "research/studies" / spec.study_id / "report.md").read_text()
    assert benchmark_spec.benchmark_id in markdown
    assert submit(seed, prune)["version"] == model.version
    elapsed = perf_counter() - start
    print(
        json.dumps(
            {
                "research_smoke_seconds": elapsed,
                "benchmark": benchmark,
                "evaluation_passed": report.passed,
            }
        )
    )
    assert elapsed < 120


def test_magnitude_ranking_and_changed_plan_refused(research):
    seed, spec, training = research
    request = PruneSpecification(
        study_id=spec.study_id, training=training, plan=PruningPlan(ranking_rule="magnitude")
    )
    response = seed.client.post(
        "/v1/research/jobs", headers=RESEARCH, json=request.model_dump(mode="json")
    )
    assert response.json()["error"]["code"] == "magnitude_ranking_unsupported"


@pytest.mark.parametrize("state", ["queued", "failed"])
def test_prune_reused_uncompleted_training_job_is_conflict(research, state):
    seed, spec, training = research
    submit(seed, spec)
    training = training.model_copy(update={"job_id": uid()})
    seed.app.state.registry.save_job(
        TrainingJob(specification=training, state=state), seed.manifest.tenant_ids
    )
    response = seed.client.post(
        "/v1/research/jobs",
        headers=RESEARCH,
        json=PruneSpecification(study_id=spec.study_id, training=training).model_dump(mode="json"),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "training_job_id_conflict"


def test_post_work_and_summary_checks_authenticate_metadata_without_decrypting_shards(
    research, monkeypatch
):
    import adaptive_llm.research.service as module

    seed, spec, _ = research
    service = seed.app.state.research
    original_instrument, original_decrypt = module.instrument, service.cipher.decrypt
    work_done = False
    shard_decryptions = 0

    def decrypt(payload, tenant, interaction, field, **kwargs):
        nonlocal shard_decryptions
        if field == "dataset":
            assert not work_done, "post-work and GET must not decrypt dataset shards"
            shard_decryptions += 1
        return original_decrypt(payload, tenant, interaction, field, **kwargs)

    def instrument(*args, **kwargs):
        nonlocal work_done
        assert kwargs["memory_limit_bytes"] == service.memory_limit_bytes
        result = original_instrument(*args, **kwargs)
        work_done = True
        return result

    monkeypatch.setattr(service.cipher, "decrypt", decrypt)
    monkeypatch.setattr(module, "instrument", instrument)
    submit(seed, spec)
    assert shard_decryptions > 0
    url = f"/v1/research/studies/{spec.study_id}"
    assert seed.client.get(url, headers=RESEARCH).status_code == 200
    actor = research_identity(seed)
    service.preconditions(spec, actor, verify_shards=False)
    service.training.policy.training["synthetic-a"] = False
    assert (
        seed.client.get(url, headers=RESEARCH).json()["error"]["code"] == "training_policy_denied"
    )
    with pytest.raises(GatewayError, match="training_policy_denied"):
        service.preconditions(spec, actor, verify_shards=False)
    service.training.policy.training["synthetic-a"] = True
    manifest = service.training.builder.get(seed.manifest.dataset_id, seed.manifest.version, actor)
    with seed.app.state.metadata.database.transaction():
        seed.app.state.metadata.approve_manifest(
            manifest.model_copy(update={"approval": DatasetApproval()})
        )
    assert (
        seed.client.get(url, headers=RESEARCH).json()["error"]["code"]
        == "dataset_approval_required"
    )
    with pytest.raises(GatewayError, match="dataset_approval_required"):
        service.preconditions(spec, actor, verify_shards=False)
    with seed.app.state.metadata.database.transaction():
        seed.app.state.metadata.approve_manifest(manifest)
    path = seed.directory / "datasets" / manifest.dataset_id / manifest.version / "manifest.mac"
    path.write_text("synthetic-tamper")
    assert (
        seed.client.get(url, headers=RESEARCH).json()["error"]["code"] == "invalid_dataset_artifact"
    )
    with pytest.raises(GatewayError, match="invalid_dataset_artifact"):
        service.preconditions(spec, actor, verify_shards=False)


def test_old_layer_statistics_require_a_new_study(research):
    seed, spec, training = research
    submit(seed, spec)
    store = seed.app.state.research.store
    path = store.path(spec.study_id) / "summary.json"
    envelope = json.loads(path.read_bytes())
    envelope["summary"]["hook_version"] = "aggregate-taylor-v1"
    envelope.pop("mac")
    envelope["mac"] = store._mac(envelope)
    path.write_text(json.dumps(envelope))
    assert (
        seed.client.get(f"/v1/research/studies/{spec.study_id}", headers=RESEARCH).status_code
        == 200
    )
    response = seed.client.post(
        "/v1/research/jobs",
        headers=RESEARCH,
        json=PruneSpecification(
            study_id=spec.study_id,
            training=training,
            plan=PruningPlan(structures=["layers"], maximum_fraction=0.5),
        ).model_dump(mode="json"),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "research_hook_version_mismatch"


def test_head_only_and_layer_pruning_load_and_preserve_manifest_integrity(research):
    seed, spec, training = research
    submit(seed, spec)
    for plan in (PruningPlan(), PruningPlan(structures=["layers"], maximum_fraction=0.5)):
        model = ModelManifest.model_validate(
            submit(
                seed,
                PruneSpecification(
                    study_id=spec.study_id,
                    plan=plan,
                    training=training.model_copy(
                        update={"job_id": uid(), "steps": 2, "checkpoint_every": 1}
                    ),
                ),
            )
        )
        path = seed.directory / model.storage_location
        provider = SpecialistProvider(model, path, seed.app.state.keyring, seed.directory)
        generated = asyncio.run(
            provider.generate(
                ProviderRequest(
                    (Message(role="user", content="SYNTHETIC structured removal"),),
                    (),
                    ResponseFormat(),
                    8,
                )
            )
        )
        assert generated.usage.output_tokens > 0
        assert provider._real.model.config.num_hidden_layers == (
            1 if plan.structures == ["layers"] else 2
        )
        changed = model.model_copy(
            update={"pruning": model.pruning.model_copy(update={"parameter_count_after": 1})}
        )
        with pytest.raises(GatewayError, match="artifact_integrity_failed"):
            SpecialistProvider(changed, path, seed.app.state.keyring, seed.directory)
        (path / "model.safetensors").write_bytes(b"synthetic-tamper")
        with pytest.raises(GatewayError, match="artifact_integrity_failed"):
            SpecialistProvider(model, path, seed.app.state.keyring, seed.directory)


def test_registry_requires_signed_hardware_benefit_and_zero_critical_failures(
    research, monkeypatch
):
    from adaptive_llm.contracts import (
        BenchmarkMeasurement,
        BenchmarkReport,
        PairedComparison,
        PromotionRequest,
    )

    seed, spec, training = research
    response = seed.client.post(
        "/v1/datasets/builds",
        headers=OPERATOR,
        json=seed.manifest.specification.model_copy(
            update={"dataset_id": "synthetic-calibration"}
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.json()
    calibration = DatasetManifest.model_validate(response.json())
    response = seed.client.post(
        f"/v1/datasets/{calibration.dataset_id}/versions/{calibration.version}/approval",
        headers=OPERATOR,
        json={"reason": "SYNTHETIC calibration approval"},
    )
    assert response.status_code == 200, response.json()
    spec = spec.model_copy(
        update={
            "calibration_dataset_id": calibration.dataset_id,
            "calibration_dataset_version": calibration.version,
        }
    )
    training = training.model_copy(
        update={"dataset_id": calibration.dataset_id, "dataset_version": calibration.version}
    )
    submit(seed, spec)
    model = ModelManifest.model_validate(
        submit(
            seed,
            PruneSpecification(
                study_id=spec.study_id,
                plan=PruningPlan(structures=["attention_heads", "mlp_channels"]),
                training=training.model_copy(update={"job_id": uid()}),
            ),
        )
    )
    evaluation = evaluate(seed, model.version)
    assert evaluation.passed
    from adaptive_llm.research.service import lineage

    assert model.datasets == [lineage(calibration)]
    assert model.pruning.evaluation_dataset == lineage(seed.manifest)
    assert evaluation.specification.dataset_version == seed.manifest.version
    envelopes = seed.app.state.registry.database.connection.execute(
        "SELECT envelope FROM outbox WHERE event_type='training.completed.v1'"
    ).fetchall()
    completed = [
        json.loads(row[0])["data"]
        for row in envelopes
        if json.loads(row[0])["data"]["job_id"] == model.training_job_id
    ]
    assert completed
    assert all(
        event["dataset_refs"] == [f"{calibration.dataset_id}/{calibration.version}"]
        for event in completed
    )
    response = seed.client.post(
        "/v1/evaluations",
        headers=OPERATOR,
        json=seed.request.model_copy(
            update={
                "evaluation_id": uid(),
                "candidate_deployment_id": model.version,
                "dataset_id": calibration.dataset_id,
                "dataset_version": calibration.version,
            }
        ).model_dump(mode="json"),
    )
    assert response.json()["error"]["code"] == "model_dataset_mismatch"
    benchmarker = seed.app.state.benchmarks
    registry = seed.app.state.registry
    actor = research_identity(seed)
    evaluation_id = evaluation.specification.evaluation_id
    assert not registry._passed(model, evaluation_id, actor)
    source = seed.app.state.research.identify_source(spec.base, actor)
    base = BenchmarkMeasurement(
        requests=8,
        successes=8,
        p50_latency_ms=10,
        p95_latency_ms=20,
        requests_per_second=50,
        peak_rss_bytes=1000,
        parameter_count=model.pruning.parameter_count_before,
        total_cost_micros=10,
        cost_per_success_micros=2,
        input_micros_per_1000_tokens=1000,
        output_micros_per_1000_tokens=2000,
    )
    measured = BenchmarkReport(
        specification=BenchmarkSpecification(
            candidate_version=model.version, evaluation_id=evaluation_id, requests=8
        ),
        tenant_ids=model.tenant_ids,
        candidate_artifact_digest=model.artifact_digest,
        teacher_version=source.version,
        teacher_artifact_digest=source.artifact_digest,
        dataset_content_digest=evaluation.dataset_content_digest,
        request_mix_digest="synthetic-gate",
        student=base.model_copy(update={"parameter_count": model.pruning.parameter_count_after}),
        teacher=base,
        quality_comparison=PairedComparison(sample_size=8, mean_delta=0, ci_lower=0, ci_upper=0),
        latency_reduction_fraction=0,
        peak_rss_reduction_fraction=0,
        cost_reduction_fraction=1,
        passed=True,
        pruning_study_id=spec.study_id,
        known_limitations=["Synthetic numerical gate fixture, not a hardware measurement."],
    )
    # A signed 'passed' bit and parameter/cost savings cannot override current hardware targets.
    benchmarker.store.publish(measured, actor)
    assert not registry._passed(model, evaluation_id, actor)
    good = measured.model_copy(
        update={
            "specification": measured.specification.model_copy(update={"benchmark_id": uid()}),
            "latency_reduction_fraction": 0.5,
            "student": measured.student.model_copy(update={"p95_latency_ms": 10}),
        }
    )
    saved = benchmarker.store.publish(good, actor)
    assert registry._passed(model, evaluation_id, actor)
    # The registry itself must bind evaluation data even if the benchmark gate accepts it.
    with monkeypatch.context() as patch:
        patch.setattr(registry, "benchmark_gate", lambda *args: True)
        patch.setattr(
            registry.evaluations,
            "get",
            lambda *args: evaluation.model_copy(
                update={
                    "specification": evaluation.specification.model_copy(
                        update={
                            "dataset_id": calibration.dataset_id,
                            "dataset_version": calibration.version,
                        }
                    ),
                    "dataset_content_digest": calibration.content_digest,
                }
            ),
        )
        assert not registry._passed(model, evaluation_id, actor)
    for update in (
        {
            "suite_results": [
                s.model_copy(update={"metrics": {**s.metrics, "critical_failures": 1}})
                if s.suite == "safety"
                else s
                for s in evaluation.suite_results
            ]
        },
        {"suite_results": [s for s in evaluation.suite_results if s.suite != "safety"]},
    ):
        assert not benchmarker.allows(model, evaluation.model_copy(update=update), actor)
    with registry.database.transaction():
        registry.database.connection.execute(
            "UPDATE benchmark_reports SET report=? WHERE benchmark_id=?",
            (
                saved.model_copy(
                    update={"candidate_artifact_digest": "synthetic-tamper"}
                ).model_dump_json(),
                saved.specification.benchmark_id,
            ),
        )
    with pytest.raises(GatewayError, match="benchmark_integrity_failed"):
        registry._passed(model, evaluation_id, actor)
    with registry.database.transaction():
        registry.database.connection.execute(
            "UPDATE benchmark_reports SET report=? WHERE benchmark_id=?",
            (
                saved.model_dump_json(),
                saved.specification.benchmark_id,
            ),
        )
    assert (
        registry.promote(
            PromotionRequest(
                model_version=model.version,
                target_state="approved",
                reason="SYNTHETIC gate verification",
                evaluation_id=evaluation_id,
            ),
            actor,
        ).state
        == "approved"
    )
