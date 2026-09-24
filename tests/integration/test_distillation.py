import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest
from test_training import OPERATOR, approve, evaluate, promote, spec_for, train

from adaptive_llm.contracts import (
    AdapterConfig,
    BenchmarkReport,
    BenchmarkSpecification,
    DatasetManifest,
    DatasetSpecification,
    Message,
    ResponseFormat,
    uid,
)
from adaptive_llm.datasets.artifacts import read_shards, read_soft_targets
from adaptive_llm.datasets.eligibility import select
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.providers import FakeProvider, ProviderRequest
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.training.lora import base_files, cpu, generate_tiny_base, libraries


def identity(seed):
    return seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )


def distillation_spec(seed, **updates):
    return DatasetSpecification.model_validate(
        {
            **seed.manifest.specification.model_dump(),
            "dataset_id": "synthetic-distillation",
            "purpose": "distillation",
            "source_dataset_id": seed.manifest.dataset_id,
            "source_dataset_version": seed.manifest.version,
            "teacher_deployment_id": "fake-foundation-local-1",
            "general_safety_fraction": 0,
            **updates,
        }
    )


def build(seed, **updates):
    response = seed.client.post(
        "/v1/datasets/builds",
        headers=OPERATOR,
        json=distillation_spec(seed, **updates).model_dump(mode="json"),
    )
    assert response.status_code == 200, response.json()
    return DatasetManifest.model_validate(response.json())


def use_dataset(seed, manifest):
    seed.manifest = manifest
    seed.request = seed.request.model_copy(
        update={
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.version,
        }
    )
    approve(seed)


def shards(seed, manifest):
    service = seed.app.state.training
    return read_shards(manifest, seed.directory, service.cipher, service.keyring)


def smoke_suites(seed):
    directory = seed.directory / "student-smoke-suites"
    for name, filename in (
        ("golden", "synthetic.jsonl"),
        ("safety", "synthetic.jsonl"),
        ("retrieval", "relevance.jsonl"),
    ):
        path = directory / name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "input": "SYNTHETIC unique case 4 unused receipt",
                    "target": "",
                    "expect_contains": ["SYNTHETIC ANSWER: No context supplied."],
                    "expect_citation": False,
                    "prohibited": ["SYNTHETIC_SECRET"],
                }
            )
            + "\n"
        )
    seed.app.state.evaluations.fixture_dir = directory


@pytest.fixture
def student_base(tmp_path):
    try:
        libraries()
    except GatewayError:
        pytest.skip("optional training group absent")
    path = Path(__file__).parents[1] / "fixtures/base-models/generated/tiny-student/seed-17"
    generate_tiny_base(path, student=True)
    return path


@pytest.fixture
def student(evaluation_seed, student_base):
    import shutil

    seed = evaluation_seed
    shutil.copytree(student_base, seed.directory / "base-models/tiny-student/seed-17")
    approve(seed)
    use_dataset(seed, build(seed))
    spec = spec_for(seed).model_copy(
        update={
            "job_type": "distillation",
            "registry_id": "synthetic-student",
            "base_model_id": "tiny-student",
            "base_model_revision": "seed-17",
            "base_model_licence": "CC0-1.0",
            "tokenizer_id": "tiny-byte-v1",
            "chat_template_version": "tiny-chat-v1",
            "max_sequence_length": 128,
            "steps": 4,
            "checkpoint_every": 2,
            "input_micros_per_1000_tokens": 1,
            "output_micros_per_1000_tokens": 1,
            "adapter_config": AdapterConfig(learning_rate=0.005, dropout=0.1),
        }
    )
    return seed, spec


def alternate_base(seed, name, layers, hidden):
    """Offline fixture with dimensions deliberately different from the first student."""
    path = seed.directory / "base-models" / name / "seed-17"
    generate_tiny_base(path)
    libs = libraries()
    config = libs.transformers.LlamaConfig.from_dict(json.loads((path / "config.json").read_text()))
    config.num_hidden_layers = layers
    config.hidden_size = hidden
    config.head_dim = hidden // config.num_attention_heads
    config.intermediate_size = hidden * 2
    with cpu(libs, 17):
        libs.transformers.LlamaForCausalLM(config).save_pretrained(path, safe_serialization=True)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["model_id"] = name
    manifest["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.iterdir()
        if p.name != "manifest.json"
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return base_files(seed.directory, name, "seed-17", "tiny-byte-v1", "tiny-chat-v1", "CC0-1.0")


def test_distillation_approval_mix_provenance_and_privacy(evaluation_seed, caplog):
    seed = evaluation_seed
    refused = seed.client.post(
        "/v1/datasets/builds",
        headers=OPERATOR,
        json=distillation_spec(seed).model_dump(mode="json"),
    )
    assert refused.status_code == 409
    approve(seed)
    original = seed.app.state.metadata.get_manifest(seed.manifest.dataset_id, seed.manifest.version)
    before = seed.app.state.database.connection.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]
    manifest = build(seed, general_safety_fraction=0.8, soft_targets=True)
    assert manifest.approval.status == "pending"
    assert manifest.distillation.mix_counts == {
        "teacher": 1,
        "general": 2,
        "abstention": 1,
        "escalation": 1,
    }
    assert manifest.distillation.soft_target_files == []
    assert manifest.distillation.judge_version == "deterministic-judge-1"
    assert manifest.distillation.generation_parameters["max_output_tokens"] == 256
    original_rows, rows = shards(seed, original), shards(seed, manifest)
    assert rows["test"] and len(rows["test"]) == len(original_rows["test"]) == 8
    for old, new in zip(original_rows["test"], rows["test"], strict=True):
        left, right = json.loads(old), json.loads(new)
        for key in ("input", "sources", "target", "interaction_id", "example_hash"):
            assert left[key] == right[key]
    assert "<<source" in json.loads(rows["test"][0])["input"]["sources"][0]
    assert (
        before
        == seed.app.state.database.connection.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]
    )
    for forbidden in ("SYNTHETIC ANSWER", "unused receipt", "private credentials"):
        assert forbidden not in manifest.model_dump_json() + caplog.text
        directory = seed.directory / "datasets" / manifest.dataset_id / manifest.version
        assert all(forbidden.encode() not in p.read_bytes() for p in directory.iterdir())
    denied = seed.client.post(
        "/v1/training/jobs",
        headers=OPERATOR,
        json=spec_for(seed)
        .model_copy(
            update={
                "job_type": "distillation",
                "dataset_id": manifest.dataset_id,
                "dataset_version": manifest.version,
            }
        )
        .model_dump(mode="json"),
    )
    assert denied.status_code == 409


def test_teacher_training_rows_keep_rag_and_current_policy_and_deletion_apply(evaluation_seed):
    seed = evaluation_seed
    approve(seed)
    # A second approved source makes the evidence-bearing rows part of its train fold.
    response = seed.client.post(
        "/v1/datasets/builds",
        headers=OPERATOR,
        json=seed.manifest.specification.model_copy(
            update={
                "dataset_id": "synthetic-context-source",
                "time_split": None,
            }
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200
    source = DatasetManifest.model_validate(response.json())
    use_dataset(seed, source)
    source = seed.app.state.metadata.get_manifest(source.dataset_id, source.version)
    original = {
        json.loads(r)["interaction_id"]: json.loads(r) for r in shards(seed, source)["train"]
    }
    manifest = build(seed)
    trained = [json.loads(r) for r in shards(seed, manifest)["train"]]
    evidence = [r for r in trained if r["sources"]]
    assert evidence
    for row in evidence:
        assert row["input"] == original[row["interaction_id"]]["input"]
        assert row["sources"] == original[row["interaction_id"]]["sources"]
        assert row["target_attempt_id"] is None
        assert (
            row["source_target_attempt_id"] == original[row["interaction_id"]]["target_attempt_id"]
        )
    use_dataset(seed, manifest)
    spec = spec_for(seed).model_copy(update={"job_type": "distillation"})
    seed.app.state.training.policy.training["synthetic-b"] = False
    assert (
        seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
        ).status_code
        == 403
    )
    seed.app.state.training.policy.training["synthetic-b"] = True
    deleted = seed.client.delete(
        f"/v1/privacy/interactions/{evidence[0]['interaction_id']}",
        headers={"Authorization": "Bearer synthetic-key-a"},
    )
    assert deleted.status_code == 204
    response = seed.client.post(
        "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "distillation_source_no_longer_eligible"


def test_teacher_must_be_approved_and_mix_cannot_silently_disappear(evaluation_seed):
    seed = evaluation_seed
    approve(seed)
    job = train(seed)
    for state in ("candidate", "evaluating", "shadow", "deprecated", "revoked"):
        model = seed.app.state.registry.get(job.model_version, identity(seed)).model_copy(
            update={"state": state}
        )
        seed.app.state.evaluation_database.connection.execute(
            "UPDATE model_versions SET data=? WHERE version=?",
            (model.model_dump_json(), model.version),
        )
        response = seed.client.post(
            "/v1/datasets/builds",
            headers=OPERATOR,
            json=distillation_spec(seed, teacher_deployment_id=job.model_version).model_dump(
                mode="json"
            ),
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "teacher_not_approved"
    response = seed.client.post(
        "/v1/datasets/builds",
        headers=OPERATOR,
        json=distillation_spec(seed, general_safety_fraction=0.2).model_dump(mode="json"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "distillation_mix_insufficient_examples"


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("judge", "teacher_judge"),
        ("hard", "teacher_validation_or_generation"),
        ("golden", "benchmark_contamination"),
        ("redaction", "build_redaction_failed"),
    ],
)
def test_teacher_targets_filtered_and_counted(evaluation_seed, failure, reason, monkeypatch):
    seed = evaluation_seed
    approve(seed)
    evaluator = seed.app.state.evaluations
    deployment = evaluator.deployments[evaluator.foundation_id]

    class Teacher(FakeProvider):
        async def generate(self, request):
            output = await super().generate(request)
            if failure == "hard":
                return replace(output, content="", finish_reason="error")
            if failure == "judge":
                return replace(output, content="SYNTHETIC incorrect unrelated answer")
            return output

    evaluator.deployments[evaluator.foundation_id] = replace(deployment, provider=Teacher())
    if failure == "golden":
        monkeypatch.setattr(
            "adaptive_llm.distillation.data.shingles", lambda _: frozenset({("same",)})
        )
    if failure == "redaction":

        def broken(*args):
            raise RuntimeError("fixed_synthetic_failure")

        monkeypatch.setattr(seed.app.state.persistence.redactor, "redact_text", broken)
    builder = seed.app.state.datasets
    spec = distillation_spec(seed, near_duplicate_threshold=0.8)
    selection = select(builder.persistence.metadata, builder.policy, spec, builder.clock())
    counts = Counter()
    result = builder.distillation_examples.build(selection.examples, spec, identity(seed), counts)
    assert counts[reason] == 1
    assert all(e.row["split"] != "train" for e in result.examples)


@pytest.mark.accelerator_free
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_student_determinism_resume_limits_and_verified_loading(student, mode):
    seed, original = student
    spec = original.model_copy(update={"student_training": mode})
    first = train(seed, spec)
    assert first.state == "succeeded", first.failure_code
    second = train(seed, spec.model_copy(update={"job_id": uid()}))
    assert first.artifact_digest == second.artifact_digest
    service = seed.app.state.training
    trainer = service.student_trainers[mode]
    original_train = trainer.train

    def interrupt(specification, dataset, directory, refs, checkpoint, check):
        def stop(reference):
            checkpoint(reference)
            if reference == "checkpoint-2":
                raise GatewayError(503, "training_interrupted")

        return original_train(specification, dataset, directory, refs, stop, check)

    trainer.train = interrupt
    retry = spec.model_copy(update={"job_id": uid()})
    failed = train(seed, retry)
    assert failed.failure_code == "training_interrupted"
    trainer.train = original_train
    resumed = train(seed, retry)
    assert resumed.artifact_digest == first.artifact_digest
    registry = seed.app.state.registry
    model = registry.get(first.model_version, identity(seed))
    assert model.student_architecture == "llama-1x16-v1"
    assert model.student_parameter_count == 10640
    assert model.distillation.teacher_parameter_count is None
    assert "teacher size unknown; size reduction not verified" in model.known_limitations
    assert model.manifest_mac_version == "2"
    assert model.adapter_architecture == ("student-full-v1" if mode == "full" else "lora-peft-v1")
    provider = SpecialistProvider(
        model, seed.directory / first.artifact_ref, service.keyring, seed.directory
    )
    output = asyncio.run(
        provider.generate(
            ProviderRequest(
                (Message(role="user", content="SYNTHETIC test"),),
                (),
                ResponseFormat(),
                4,
            )
        )
    )
    assert 0 < output.usage.output_tokens <= 4
    trainer.memory_limit_bytes = 1
    assert (
        train(seed, spec.model_copy(update={"job_id": uid()})).failure_code
        == "training_memory_limit"
    )


def benchmark(seed, job, evaluation, **updates):
    spec = BenchmarkSpecification(
        candidate_version=job.model_version,
        evaluation_id=evaluation.specification.evaluation_id,
        **updates,
    )
    response = seed.client.post(
        "/v1/benchmarks/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
    )
    assert response.status_code == 200, response.json()
    return BenchmarkReport.model_validate(response.json())


@pytest.mark.accelerator_free
def test_unknown_teacher_size_accepts_arbitrary_student_geometry(student):
    seed, spec = student
    base = alternate_base(seed, "synthetic-larger-student", layers=3, hidden=40)
    job = train(seed, spec.model_copy(update={"base_model_id": "synthetic-larger-student"}))
    assert job.state == "succeeded", job.failure_code
    model = seed.app.state.registry.get(job.model_version, identity(seed))
    assert model.student_architecture == "llama-3x40-v1"
    assert model.student_parameter_count == base.parameter_count > 35168
    assert model.distillation.teacher_parameter_count is None
    assert "teacher size unknown; size reduction not verified" in model.known_limitations


@pytest.mark.smoke
@pytest.mark.accelerator_free
@pytest.mark.parametrize("evaluation_seed", ["no_context"], indirect=True)
def test_student_cpu_smoke_and_benchmark_promotion(student):
    seed, spec = student
    start = perf_counter()
    job = train(seed, spec.model_copy(update={"steps": 500, "checkpoint_every": 100}))
    assert job.state == "succeeded", job.failure_code
    smoke_suites(seed)
    assert evaluate(seed).passed
    report = evaluate(seed, job.model_version)
    assert report.passed, [(g.gate, g.reason) for g in decisions(report) if not g.passed]
    assert report.segment_comparisons["distillation"].sample_size == 8
    assert {s.suite for s in report.suite_results} == {
        "golden",
        "held_out",
        "safety",
        "retrieval",
        "performance",
    }
    blocked = seed.client.post(
        f"/v1/models/{job.model_version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": job.model_version,
            "target_state": "approved",
            "reason": "synthetic",
            "evaluation_id": report.specification.evaluation_id,
        },
    )
    assert blocked.status_code == 409
    first = benchmark(seed, job, report, requests=16)
    second = benchmark(seed, job, report, requests=16)
    assert first.passed and second.passed
    assert first.request_mix_digest == second.request_mix_digest
    assert first.student.total_cost_micros == second.student.total_cost_micros
    assert first.quality_comparison == second.quality_comparison
    # Scheduler noise is expected; repeated warm measurements must remain within a bounded factor.
    assert 0.2 < first.student.p95_latency_ms / second.student.p95_latency_ms < 5
    assert seed.client.get(
        f"/v1/benchmarks/jobs/{first.specification.benchmark_id}", headers=OPERATOR
    ).json() == first.model_dump(mode="json")
    for headers, status in [({}, 401), ({"Authorization": "Bearer synthetic-key-a"}, 403)]:
        assert (
            seed.client.get(
                f"/v1/benchmarks/jobs/{first.specification.benchmark_id}", headers=headers
            ).status_code
            == status
        )
    database = seed.app.state.evaluation_database
    signed = database.connection.execute(
        "SELECT report FROM benchmark_reports WHERE benchmark_id=?",
        (first.specification.benchmark_id,),
    ).fetchone()[0]
    database.connection.execute(
        "UPDATE benchmark_reports SET report=? WHERE benchmark_id=?",
        (
            first.model_copy(update={"candidate_artifact_digest": "tampered"}).model_dump_json(),
            first.specification.benchmark_id,
        ),
    )
    refused = seed.client.post(
        f"/v1/models/{job.model_version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": job.model_version,
            "target_state": "approved",
            "reason": "synthetic",
            "evaluation_id": report.specification.evaluation_id,
        },
    )
    assert refused.status_code == 409
    database.connection.execute(
        "UPDATE benchmark_reports SET report=? WHERE benchmark_id=?",
        (signed, first.specification.benchmark_id),
    )
    assert (
        promote(seed, job.model_version, "approved", report.specification.evaluation_id).state
        == "approved"
    )
    elapsed = perf_counter() - start
    assert elapsed < 90
    print(
        f"student smoke: elapsed_s={elapsed:.3f}; benchmark={first.model_dump_json()}; "
        f"training_peak_rss={job.resource_usage.peak_memory_bytes}"
    )


@pytest.mark.accelerator_free
def test_worse_student_runs_five_suites_and_teacher_comparison(student):
    seed, spec = student
    evaluate(seed)
    job = train(seed, spec)
    report = evaluate(seed, job.model_version, performance_requests=4)
    assert not report.passed
    assert not next(g.passed for g in decisions(report) if g.gate == "distillation_non_inferiority")
    measured = benchmark(seed, job, report, requests=8)
    assert not measured.passed


@pytest.mark.accelerator_free
@pytest.mark.parametrize("evaluation_seed", ["no_context"], indirect=True)
def test_real_teacher_soft_targets_are_encrypted_safetensors_and_train(
    student_base, evaluation_seed
):
    import shutil

    from adaptive_llm.training.lora import LoraTrainer

    seed = evaluation_seed
    approve(seed)
    generate_tiny_base(seed.directory / "base-models/tiny/seed-17")
    shutil.copytree(student_base, seed.directory / "base-models/tiny-student/seed-17")
    service = seed.app.state.training
    service.trainer = LoraTrainer(seed.directory, service.cipher, service.keyring)
    teacher_spec = spec_for(seed).model_copy(
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
    teacher = train(seed, teacher_spec)
    smoke_suites(seed)
    evaluate(seed)
    report = evaluate(seed, teacher.model_version)
    assert report.passed
    promote(seed, teacher.model_version, "approved", report.specification.evaluation_id)
    manifest = build(seed, teacher_deployment_id=teacher.model_version, soft_targets=True)
    use_dataset(seed, manifest)
    manifest = seed.app.state.metadata.get_manifest(manifest.dataset_id, manifest.version)
    shards(seed, manifest)
    soft = read_soft_targets(manifest, seed.directory, service.cipher)
    assert len(soft) == 1
    tensors = libraries().tensors.load(next(iter(soft.values())))
    assert set(tensors) == {"target_ids", "log_probs"}
    assert tensors["log_probs"].shape[1] == 259
    assert manifest.distillation.teacher_parameter_count > 10_000
    spec = teacher_spec.model_copy(
        update={
            "job_id": uid(),
            "job_type": "distillation",
            "base_model_id": "tiny-student",
            "dataset_id": manifest.dataset_id,
            "dataset_version": manifest.version,
            "steps": 4,
            "checkpoint_every": 2,
        }
    )
    job = train(seed, spec)
    assert job.state == "succeeded", job.failure_code
    repeated = train(seed, spec.model_copy(update={"job_id": uid()}))
    assert repeated.artifact_digest == job.artifact_digest
    no_kl = train(seed, spec.model_copy(update={"job_id": uid(), "soft_target_weight": 0}))
    losses = [
        json.loads((seed.directory / j.artifact_ref / "training_report.json").read_text())[
            "loss_curve"
        ]
        for j in (job, no_kl)
    ]
    assert losses[0] != losses[1]
    equal = train(seed, spec.model_copy(update={"job_id": uid(), "base_model_id": "tiny"}))
    assert equal.state == "failed" and equal.failure_code == "student_must_be_smaller"
    base = alternate_base(seed, "synthetic-alternate-student", layers=3, hidden=8)
    different = train(
        seed,
        spec.model_copy(
            update={
                "job_id": uid(),
                "base_model_id": "synthetic-alternate-student",
            }
        ),
    )
    assert different.state == "succeeded", different.failure_code
    model = service.registry.get(different.model_version, identity(seed))
    assert model.student_architecture == "llama-3x8-v1"
    assert (
        model.student_parameter_count
        == base.parameter_count
        < model.distillation.teacher_parameter_count
    )
    assert "teacher size unknown; size reduction not verified" not in model.known_limitations
    path = seed.directory / "datasets" / manifest.dataset_id / manifest.version
    tensor_file = next(path.glob("*.safetensors.enc"))
    tensor_file.write_bytes(tensor_file.read_bytes() + b"tamper")
    with pytest.raises(GatewayError, match="invalid_dataset_artifact"):
        shards(seed, manifest)
