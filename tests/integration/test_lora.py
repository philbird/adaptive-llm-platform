import asyncio
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import perf_counter

import pytest
from conftest import EvaluationSeed, wait_training
from fastapi.testclient import TestClient
from test_training import OPERATOR, approve, evaluate, promote, spec_for, train

from adaptive_llm.app import create_app
from adaptive_llm.contracts import AdapterConfig, Message, ResponseFormat, uid
from adaptive_llm.evaluation.gate import decisions
from adaptive_llm.gateway.identity import GatewayError
from adaptive_llm.providers import ProviderRequest
from adaptive_llm.providers.specialist import SpecialistProvider
from adaptive_llm.training.lora import LoraTrainer, generate_tiny_base, libraries

pytestmark = pytest.mark.accelerator_free


@pytest.fixture(scope="session")
def tiny_base():
    try:
        libraries()
    except GatewayError:
        pytest.skip("optional training group is absent; install the locked training extra")
    directory = Path(__file__).parents[1] / "fixtures/base-models/generated/tiny/seed-17"
    generate_tiny_base(directory)
    return directory


@pytest.fixture
def real(evaluation_seed: EvaluationSeed, tiny_base: Path, monkeypatch):
    seed = evaluation_seed
    shutil.copytree(tiny_base, seed.directory / "base-models/tiny/seed-17")
    approve(seed)
    service = seed.app.state.training
    service.trainer = LoraTrainer(seed.directory, service.cipher, service.keyring)

    def offline(*args, **kwargs):
        raise AssertionError("model_network_access_forbidden")

    monkeypatch.setattr("socket.socket.connect", offline)
    spec = spec_for(seed).model_copy(
        update={
            "base_model_id": "tiny",
            "base_model_revision": "seed-17",
            "base_model_licence": "CC0-1.0",
            "tokenizer_id": "tiny-byte-v1",
            "chat_template_version": "tiny-chat-v1",
            "max_sequence_length": 128,
            "steps": 4,
            "checkpoint_every": 2,
            "adapter_config": AdapterConfig(rank=4, alpha=8, dropout=0.1),
        }
    )
    return seed, spec


def test_real_cancel_checkpoints_between_steps_and_never_registers(real):
    seed, spec = real
    entered, release = Event(), Event()
    service = seed.app.state.training
    delegate = service.trainer

    class PausedTrainer:
        architecture = delegate.architecture

        def train(self, specification, dataset, directory, refs, checkpoint, check):
            calls = 0

            def pause():
                nonlocal calls
                calls += 1
                if calls == 2:
                    entered.set()
                    assert release.wait(5)
                check()

            return delegate.train(specification, dataset, directory, refs, checkpoint, pause)

    service.trainer = PausedTrainer()
    try:
        response = seed.client.post(
            "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
        )
        assert response.json()["state"] == "queued"
        assert entered.wait(5)
        assert seed.client.post(f"/v1/training/jobs/{spec.job_id}/cancel", headers=OPERATOR).json()[
            "cancel_requested"
        ]
    finally:
        release.set()
    job = wait_training(seed.client, spec.job_id, OPERATOR)
    assert job.state == "cancelled" and job.checkpoint_refs == ["checkpoint-1"]
    assert job.artifact_ref is None
    path = (
        seed.directory
        / "models"
        / spec.registry_id
        / f".{job.model_version}.training/checkpoint-1/training_report.json"
    )
    assert json.loads(path.read_bytes())["steps"] == 1


def test_settings_select_lora_and_enforce_memory_limit(real):
    seed, spec = real
    seed.app.state.training.stop()
    app = create_app(
        replace(
            seed.app.state.settings,
            trainer=None,
            training_backend="lora",
            training_memory_limit_bytes=1,
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/training/jobs", headers=OPERATOR, json=spec.model_dump(mode="json")
        )
        assert response.json()["state"] == "queued"
        job = wait_training(client, spec.job_id, OPERATOR)
        assert job.failure_code == "training_memory_limit"


def model_for(seed, job):
    identity = seed.app.state.authenticator.authenticate(
        OPERATOR["Authorization"], "synthetic-operator"
    )
    return seed.app.state.registry.get(job.model_version, identity)


def test_real_determinism_resume_privacy_and_generation(real):
    seed, spec = real
    first = train(seed, spec)
    assert first.state == "succeeded", first.failure_code
    second = train(seed, spec.model_copy(update={"job_id": uid()}))
    assert second.state == "succeeded", second.failure_code
    assert first.artifact_digest == second.artifact_digest
    paths = [seed.directory / j.artifact_ref for j in (first, second)]
    reports = [json.loads((p / "training_report.json").read_bytes()) for p in paths]
    assert reports[0]["loss_curve"] == reports[1]["loss_curve"]
    assert len(reports[0]["loss_curve"]) == 4
    artifact_names = {
        "adapter.safetensors",
        "merged.safetensors",
        "adapter_config.json",
        "training_report.json",
    }
    for name in artifact_names:
        assert (paths[0] / name).read_bytes() == (paths[1] / name).read_bytes()
    assert reports[0]["tokens_seen"] > 0
    assert set(reports[0]) == {
        "steps",
        "examples",
        "loss_curve",
        "tokens_seen",
        "base_manifest_digest",
        "binding",
        "adapter_digest",
    }
    archives = []
    for job, path in zip((first, second), paths, strict=True):
        assert {p.name for p in path.iterdir()} == artifact_names
        assert set(model_for(seed, job).artifact_hashes) == artifact_names
        assert job.resource_usage.peak_memory_bytes > 0 and job.resource_usage.wall_seconds > 0
        assert job.resource_usage.artifact_bytes == sum(p.stat().st_size for p in path.iterdir())
        archive = seed.directory / "models" / spec.registry_id / ".checkpoints" / job.model_version
        assert {p.name for p in archive.iterdir()} == set(job.checkpoint_refs)
        archives.append(archive)
    for path in [*paths, *archives]:
        for file in path.rglob("*"):
            if file.is_file():
                assert file.suffix in {".safetensors", ".json"}
                for forbidden in (
                    b"SYNTHETIC unique",
                    b"SYNTHETIC ANSWER",
                    b"No context supplied",
                    b"SYNTHETIC_PRIVATE_NOTE",
                ):
                    assert forbidden not in file.read_bytes()
    service = seed.app.state.training
    original = service.trainer

    class InterruptOnce:
        architecture = original.architecture

        def train(self, specification, dataset, directory, refs, checkpoint, check):
            def interrupt(reference):
                checkpoint(reference)
                if reference == "checkpoint-2":
                    raise GatewayError(503, "training_interrupted")

            return original.train(specification, dataset, directory, refs, interrupt, check)

    service.trainer = InterruptOnce()
    retry_spec = spec.model_copy(update={"job_id": uid()})
    interrupted = train(seed, retry_spec)
    assert interrupted.failure_code == "training_interrupted"
    assert interrupted.checkpoint_refs == ["checkpoint-2"]
    service.trainer = LoraTrainer(seed.directory, service.cipher, service.keyring)
    resumed = train(seed, retry_spec)
    assert resumed.state == "succeeded", resumed.failure_code
    assert resumed.artifact_digest == first.artifact_digest
    resumed_path = seed.directory / resumed.artifact_ref
    for name in artifact_names:
        assert (resumed_path / name).read_bytes() == (paths[0] / name).read_bytes()
    assert (
        json.loads((resumed_path / "training_report.json").read_bytes())["loss_curve"]
        == reports[0]["loss_curve"]
    )
    model = model_for(seed, first)
    # The loader uses the explicit data directory even when the verified export is relocated.
    relocated = seed.directory / "relocated-export"
    shutil.copytree(paths[0], relocated)
    provider = SpecialistProvider(model, relocated, service.keyring, seed.directory)
    request = ProviderRequest(
        (Message(role="user", content="SYNTHETIC generation"),), (), ResponseFormat(), 5
    )
    one, two = [asyncio.run(provider.generate(request)) for _ in range(2)]
    assert one.content == two.content and one.usage == two.usage
    assert one.usage.source == "provider_reported" and one.usage.tokenizer == spec.tokenizer_id
    assert 0 < one.usage.output_tokens <= 5 and one.finish_reason in {"stop", "length"}


@pytest.mark.parametrize("limit", ["memory", "time"])
def test_real_resource_limits(real, limit):
    seed, spec = real
    service = seed.app.state.training
    if limit == "memory":
        service.trainer.memory_limit_bytes = 1
    else:
        service.trainer.time_limit_seconds = 0
    failed = train(seed, spec)
    assert failed.state == "failed"
    assert failed.failure_code == (
        "training_memory_limit" if limit == "memory" else "training_interrupted"
    )
    if limit == "time":
        assert failed.checkpoint_refs == ["checkpoint-0"]
        service.trainer.time_limit_seconds = 300
        assert train(seed, spec).state == "succeeded"
    else:
        assert not failed.checkpoint_refs


def test_base_and_checkpoint_tampering_are_refused(real):
    seed, spec = real
    path = seed.directory / "base-models/tiny/seed-17/model.safetensors"
    original = path.read_bytes()
    path.write_bytes(original + b"tampered")
    assert train(seed, spec).failure_code == "invalid_base_model"
    path.write_bytes(original)
    seed.app.state.training.trainer.time_limit_seconds = 0
    job = train(seed, spec)
    path = (
        seed.directory
        / "models"
        / spec.registry_id
        / f".{job.model_version}.training/checkpoint-0/adapter.safetensors"
    )
    path.write_bytes(path.read_bytes() + b"tampered")
    seed.app.state.training.trainer.time_limit_seconds = 300
    assert train(seed, spec).failure_code == "checkpoint_integrity_failed"


def test_real_adapter_runs_all_suites_and_failures_block_approval(real):
    seed, spec = real
    evaluate(seed)
    # A random tiny model trained for four steps cannot pass real behavioral assertions.
    job = train(seed, spec)
    assert job.state == "succeeded"
    report = evaluate(seed, job.model_version, performance_requests=4)
    assert {s.suite for s in report.suite_results} == {
        "golden",
        "held_out",
        "safety",
        "retrieval",
        "performance",
    }
    assert report.candidate_artifact_digest == job.artifact_digest
    assert not report.passed
    response = seed.client.post(
        f"/v1/models/{job.model_version}/promotion-requests",
        headers=OPERATOR,
        json={
            "model_version": job.model_version,
            "target_state": "approved",
            "reason": "synthetic gate check",
            "evaluation_id": report.specification.evaluation_id,
        },
    )
    assert response.status_code == 409


@pytest.mark.smoke
@pytest.mark.parametrize("evaluation_seed", ["no_context"], indirect=True)
def test_real_cpu_smoke(real):
    seed, spec = real
    start = perf_counter()
    # A deliberately narrow no-context task measures lifecycle mechanics, not general quality.
    spec = spec.model_copy(
        update={
            "steps": 500,
            "checkpoint_every": 50,
            "adapter_config": AdapterConfig(
                target_modules=["q_proj", "v_proj", "lm_head"],
                rank=8,
                alpha=16,
                learning_rate=0.005,
            ),
        }
    )
    job = train(seed, spec)
    assert job.state == "succeeded", job.failure_code
    provider = SpecialistProvider(
        model_for(seed, job),
        seed.directory / job.artifact_ref,
        seed.app.state.keyring,
        seed.directory,
    )
    request = ProviderRequest(
        (Message(role="user", content="SYNTHETIC unique case 4 unused receipt"),),
        (),
        ResponseFormat(),
        64,
    )
    result = asyncio.run(provider.generate(request))
    assert result.content == "SYNTHETIC ANSWER: No context supplied."
    assert result.finish_reason == "stop"
    for index in range(1, 9):
        check = asyncio.run(
            provider.generate(
                ProviderRequest(
                    (
                        Message(
                            role="user", content=f"SYNTHETIC unique case {index} unused receipt"
                        ),
                    ),
                    (),
                    ResponseFormat(),
                    256,
                )
            )
        )
        assert check.content == "SYNTHETIC ANSWER: No context supplied.", index
    fixture_dir = seed.directory / "smoke-suites"
    for name, filename in (
        ("golden", "synthetic.jsonl"),
        ("safety", "synthetic.jsonl"),
        ("retrieval", "relevance.jsonl"),
    ):
        path = fixture_dir / name / filename
        path.parent.mkdir(parents=True)
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
    seed.app.state.evaluations.fixture_dir = fixture_dir
    baseline = evaluate(seed)
    assert baseline.passed
    report = evaluate(seed, job.model_version)
    assert report.passed, [(g.gate, g.reason) for g in decisions(report) if not g.passed]
    promote(seed, job.model_version, "approved", report.specification.evaluation_id)
    assert promote(seed, job.model_version, "shadow").state == "shadow"
    elapsed = perf_counter() - start
    assert elapsed < 60
    digest = hashlib.sha256(
        (seed.directory / job.artifact_ref / "adapter.safetensors").read_bytes()
    ).hexdigest()
    report_data = json.loads(
        (seed.directory / job.artifact_ref / "training_report.json").read_bytes()
    )
    print(
        f"real LoRA smoke: elapsed_s={elapsed:.3f}; steps=500; adapter_sha256={digest}; "
        f"tokens={report_data['tokens_seen']}; "
        f"peak_rss_bytes={job.resource_usage.peak_memory_bytes}; "
        f"training_wall_seconds={job.resource_usage.wall_seconds}"
    )
