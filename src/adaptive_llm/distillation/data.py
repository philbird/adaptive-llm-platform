"""Teacher outputs stay in memory until filtered, redacted and encrypted by the builder."""

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import replace
from typing import Any

from adaptive_llm.contracts import (
    DatasetManifest,
    DatasetSpecification,
    DistillationLineage,
    EvaluationSpecification,
)
from adaptive_llm.datasets.artifacts import read_shards
from adaptive_llm.datasets.builder import DistilledExamples, LocalDatasetBuilder
from adaptive_llm.datasets.construction import BuiltExample
from adaptive_llm.datasets.curation import curate, golden_texts, shingles, similar
from adaptive_llm.datasets.eligibility import Example, select
from adaptive_llm.evaluation.data import HeldOutRow, LocalDatasetReader, SyntheticItem
from adaptive_llm.evaluation.judge import JudgeInput
from adaptive_llm.evaluation.runner import isolated_persistence
from adaptive_llm.evaluation.service import EvaluationDeployment, LocalEvaluator
from adaptive_llm.evaluation.suites.scoring import citations
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.providers import SoftTargetProvider


def teacher_deployment(
    evaluator: LocalEvaluator,
    deployment_id: str,
    identity: Identity,
) -> EvaluationDeployment:
    if deployment_id != evaluator.foundation_id:
        if evaluator.registry is None:
            raise GatewayError(409, "teacher_unavailable")
        model = evaluator.registry.get(deployment_id, identity)
        if model.state not in {"approved", "canary", "production"}:
            raise GatewayError(409, "teacher_not_approved")
        if model.adapter_architecture == "router-logistic-v1":
            raise GatewayError(409, "teacher_not_generative")
    return evaluator._deployment(deployment_id, identity)


def teacher_digest(deployment: EvaluationDeployment) -> str:
    return (
        deployment.artifact_digest
        or hashlib.sha256(deployment.manifest.model_dump_json().encode()).hexdigest()
    )


class TeacherExamples:
    def __init__(self, builder: LocalDatasetBuilder, evaluator: LocalEvaluator) -> None:
        self.builder, self.evaluator = builder, evaluator

    def source(self, spec: DatasetSpecification, identity: Identity) -> DatasetManifest:
        if spec.source_dataset_id is None or spec.source_dataset_version is None:
            raise GatewayError(422, "distillation_source_required")
        source = self.builder.get(spec.source_dataset_id, spec.source_dataset_version, identity)
        if source.purpose != "adapter_training" or source.approval.status != "approved":
            raise GatewayError(409, "approved_adapter_source_required")
        if set(source.tenant_ids) != set(spec.tenant_ids):
            raise GatewayError(409, "distillation_tenants_mismatch")
        for tenant in source.tenant_ids:
            trusted = replace(identity, tenant_id=tenant, application_ids=frozenset({"training"}))
            policy = self.builder.policy.decide(trusted, "training")
            if not policy.processing_allowed or not policy.training_allowed:
                raise GatewayError(403, "training_policy_denied")
            if policy.policy_version != spec.eligibility_policy_version:
                raise GatewayError(409, "eligibility_policy_version_mismatch")
        return source

    def validate(self, specification: DatasetSpecification, identity: Identity) -> None:
        self.source(specification, identity)
        teacher_deployment(self.evaluator, specification.teacher_deployment_id or "", identity)

    def validate_dataset(self, manifest: DatasetManifest, identity: Identity) -> None:
        """Recheck source deletion, retention and current per-application policy at execution."""
        self.validate(manifest.specification, identity)
        builder = self.builder
        selection = select(
            builder.persistence.metadata, builder.policy, manifest.specification, builder.clock()
        )
        eligible = {
            (e.interaction.tenant_id, e.interaction.interaction_id)
            for e in selection.examples
            if e.interaction.environment == identity.environment
        }
        rows = read_shards(
            manifest, builder.data_dir, builder.persistence.cipher, builder.persistence.keyring
        )
        if any(
            (row["tenant_id"], row["interaction_id"]) not in eligible
            for split in rows.values()
            for raw in split
            if (row := json.loads(raw))
        ):
            raise GatewayError(409, "distillation_source_no_longer_eligible")

    def build(
        self,
        examples: list[Example],
        specification: DatasetSpecification,
        identity: Identity,
        exclusions: Counter[str],
    ) -> DistilledExamples:
        try:
            return asyncio.run(self._build(examples, specification, identity, exclusions))
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(503, "distillation_build_failed") from None

    async def _build(
        self,
        examples: list[Example],
        spec: DatasetSpecification,
        identity: Identity,
        exclusions: Counter[str],
    ) -> DistilledExamples:
        source = self.source(spec, identity)
        builder, evaluator = self.builder, self.evaluator
        persistence = builder.persistence
        rows = read_shards(source, builder.data_dir, persistence.cipher, persistence.keyring)
        teacher = teacher_deployment(evaluator, spec.teacher_deployment_id or "", identity)
        eligible = {(e.interaction.tenant_id, e.interaction.interaction_id): e for e in examples}
        reader = LocalDatasetReader(
            builder, builder.data_dir, persistence.cipher, persistence.keyring
        )
        evaluation = EvaluationSpecification(
            candidate_deployment_id=spec.teacher_deployment_id or "",
            baseline_deployment_id=None,
            dataset_id=source.dataset_id,
            dataset_version=source.version,
            suites=["held_out"],
        )
        golden = golden_texts(builder.golden_dir)
        # Compare targets alone too: a copied golden answer cannot hide in a long input.
        golden_targets = [
            json.loads(line)["target"]
            for path in builder.golden_dir.glob("*.jsonl")
            if path.name != "distillation-training.jsonl"
            for line in path.read_text().splitlines()
        ]
        accepted: list[BuiltExample] = []
        soft: dict[str, bytes] = {}
        with isolated_persistence(persistence) as scratch:
            runner = evaluator._runner(teacher.manifest.model_deployment_id, identity, scratch)
            for split, raw_rows in rows.items():
                for raw in raw_rows:
                    row: dict[str, Any] = json.loads(raw)
                    original = eligible.get((row["tenant_id"], row["interaction_id"]))
                    if original is None:
                        exclusions["source_no_longer_eligible"] += 1
                        continue
                    row["source_example_hash"] = row["example_hash"]
                    if split == "train":
                        case = reader._case(HeldOutRow.model_validate(row), evaluation)
                        case = replace(
                            case,
                            request=case.request.model_copy(
                                update={
                                    "max_output_tokens": spec.teacher_max_output_tokens,
                                }
                            ),
                        )
                        outcome = await runner.run(case)
                        if outcome.response is None:
                            exclusions["teacher_validation_or_generation"] += 1
                            continue
                        if outcome.response.finish_reason != "stop":
                            exclusions["teacher_incomplete"] += 1
                            continue
                        target = outcome.response.content
                        precision, recall = citations(case, outcome)
                        score = (
                            evaluator.judge.score(
                                JudgeInput(
                                    target,
                                    (case.target,),
                                    (),
                                    precision == recall == 1,
                                )
                            )
                            / 5
                        )
                        if score < spec.teacher_minimum_score:
                            exclusions["teacher_judge"] += 1
                            continue
                        if any(
                            shingles(target) == shingles(g)
                            or similar(shingles(target), shingles(g), spec.near_duplicate_threshold)
                            for g in golden_targets
                            if g
                        ):
                            exclusions["benchmark_contamination"] += 1
                            continue
                        try:
                            redacted, _ = persistence.redactor.redact_text(target, original.policy)
                        except Exception:
                            exclusions["build_redaction_failed"] += 1
                            continue
                        if redacted != target or not target.strip():
                            exclusions["teacher_redaction_changed"] += 1
                            continue
                        row["target"] = target
                        row["source_target_attempt_id"] = row.get("target_attempt_id")
                        row["source_target_feedback_id"] = row.get("target_feedback_id")
                        row["target_attempt_id"] = None
                        row["target_feedback_id"] = None
                        row["target_source"] = "teacher"
                        row["distillation_kind"] = "teacher"
                        row["example_hash"] = persistence.keyring.dataset_hash(
                            json.dumps(
                                [row["input"], target],
                                sort_keys=True,
                            )
                        )
                    item = BuiltExample(
                        row["interaction_id"],
                        row["tenant_id"],
                        row.get("subject_id_pseudonymous"),
                        tuple(row["document_families"]),
                        original.interaction.started_at,
                        "teacher" if split == "train" else row["target_source"],
                        row["labels"]["language"],
                        row["example_hash"],
                        "\n".join(
                            [
                                *(m["content"] for m in row["input"]["messages"]),
                                *row["input"]["sources"],
                                row["target"],
                            ]
                        ),
                        row,
                    )
                    accepted.append(item)
        train = curate(
            [e for e in accepted if e.row["split"] == "train"],
            golden,
            spec.near_duplicate_threshold,
            exclusions,
        )
        # Held-out folds are immutable source targets, never teacher-generated or reshuffled.
        held_out = [e for e in accepted if e.row["split"] != "train"]
        if spec.soft_targets and isinstance(teacher.provider, SoftTargetProvider):
            for item in train:
                tensors = await teacher.provider.soft_targets(json.dumps(item.row).encode())
                if tensors is not None:
                    soft[item.exact_hash] = tensors
        mixed, mix_digest = self._mix(train, spec, golden)
        parameters: int | None = None
        tokenizer: str | None = None
        template: str | None = None
        if teacher.registered_version and evaluator.registry is not None:
            model = evaluator.registry.get(teacher.registered_version, identity)
            if model.adapter_architecture in {"lora-peft-v1", "student-full-v1"}:
                from adaptive_llm.training.lora import base_files

                parameters = base_files(
                    builder.data_dir,
                    model.base_model_id,
                    model.base_model_revision,
                    model.tokenizer_id,
                    model.chat_template_version,
                    model.base_model_licence,
                ).parameter_count
                tokenizer, template = model.tokenizer_id, model.chat_template_version
        return DistilledExamples(
            [*train, *mixed, *held_out],
            DistillationLineage(
                source_dataset_id=source.dataset_id,
                source_dataset_version=source.version,
                source_content_digest=source.content_digest,
                teacher_deployment_id=spec.teacher_deployment_id or "",
                teacher_version=teacher.version,
                teacher_artifact_digest=teacher_digest(teacher),
                teacher_parameter_count=parameters,
                tokenizer_id=tokenizer,
                chat_template_version=template,
                generation_parameters={
                    "max_output_tokens": spec.teacher_max_output_tokens,
                    "do_sample": False,
                },
                judge_version=evaluator.judge.version,
                rubric_version=evaluator.judge.rubric_version,
                minimum_score=spec.teacher_minimum_score,
                requested_mix_fraction=spec.general_safety_fraction,
                mix_counts={},
                mix_content_digest=mix_digest,
            ),
            soft,
        )

    def _mix(
        self,
        train: list[BuiltExample],
        spec: DatasetSpecification,
        golden: list[str],
    ) -> tuple[list[BuiltExample], str]:
        paths = [
            self.builder.golden_dir.parent / name / "distillation-training.jsonl"
            for name in ("golden", "safety")
        ]
        content = b"".join(p.read_bytes() for p in paths)
        fixtures = [
            SyntheticItem.model_validate_json(line)
            for p in paths
            for line in p.read_text().splitlines()
        ]
        count = round(
            len(train) * spec.general_safety_fraction / (1 - spec.general_safety_fraction)
        )
        if train and spec.general_safety_fraction > 0 and count < len(fixtures):
            raise GatewayError(422, "distillation_mix_insufficient_examples")
        mixed: list[BuiltExample] = []
        for index in range(count):
            item = fixtures[index % len(fixtures)]
            original = train[index % len(train)]
            text = item.input + "\n" + item.target
            if any(
                shingles(text) == shingles(g)
                or similar(shingles(text), shingles(g), spec.near_duplicate_threshold)
                for g in golden
            ):
                raise GatewayError(409, "distillation_mix_contaminated")
            row = dict(original.row)
            digest = self.builder.persistence.keyring.dataset_hash(text + str(index))
            row.update(
                {
                    "input": {"messages": [{"role": "user", "content": item.input}], "sources": []},
                    "target": item.target,
                    "target_source": "synthetic_mix",
                    "example_hash": digest,
                    "sources": [],
                    "document_families": [],
                    "distillation_kind": item.category,
                    "fixture_digest": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            mixed.append(
                replace(
                    original, exact_hash=digest, text=text, row=row, target_source="synthetic_mix"
                )
            )
        return mixed, hashlib.sha256(content).hexdigest()
