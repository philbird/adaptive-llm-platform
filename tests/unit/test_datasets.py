import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from adaptive_llm.app import Settings
from adaptive_llm.contracts import DatasetSpecification, Interaction, SourceWindow, TimeSplit
from adaptive_llm.datasets.builder import code_revision
from adaptive_llm.datasets.construction import BuiltExample, Excluded
from adaptive_llm.datasets.curation import curate, golden_texts, shingles, similar, split_examples
from adaptive_llm.datasets.eligibility import Example, select
from adaptive_llm.datasets.sources import LocalSourceResolver

if TYPE_CHECKING:
    from conftest import DatasetSeed

AT = datetime(2026, 9, 23, tzinfo=UTC)


def spec(**changes: object) -> DatasetSpecification:
    base = DatasetSpecification(
        dataset_id="synthetic",
        tenant_ids=["synthetic-a"],
        source_window=SourceWindow(start=AT, end=AT + timedelta(days=5)),
        eligibility_policy_version="synthetic",
    )
    return DatasetSpecification.model_validate({**base.model_dump(), **changes})


def example(
    index: int,
    family: Sequence[str] = (),
    subject: str | None = None,
    text: str | None = None,
    digest: str | None = None,
) -> BuiltExample:
    return BuiltExample(
        str(index),
        "synthetic-a",
        subject,
        tuple(family),
        AT + timedelta(hours=index),
        "production_output",
        "en",
        digest or str(index),
        text or f"SYNTHETIC example {index}",
        {},
    )


def test_joint_components_prevent_transitive_family_and_subject_leakage() -> None:
    examples = [
        example(0, ["a"], "one"),
        example(1, ["b"], "one"),
        example(2, ["b"], "two"),
        example(3, ["c"], "two"),
    ]
    examples += [example(i, [f"family-{i}"], f"subject-{i}") for i in range(4, 50)]
    splits = split_examples(examples, spec(seed=34))
    assert splits == split_examples(list(reversed(examples)), spec(seed=34))
    assert all(splits.values())
    assignments = {e.interaction_id: split for split, rows in splits.items() for e in rows}
    assert len({assignments[str(i)] for i in range(4)}) == 1
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            for key in [row.subject, *row.families]:
                assert seen.setdefault(key, split) == split


def test_time_splits_move_complete_crossing_components_later() -> None:
    examples = [example(0, ["shared"]), example(48, ["shared"]), example(1), example(25)]
    splits = split_examples(
        examples,
        spec(
            time_split=TimeSplit(
                train_end=AT + timedelta(days=1), validation_end=AT + timedelta(days=2)
            )
        ),
    )
    assert {e.interaction_id for e in splits["test"]} == {"0", "48"}
    assert [e.interaction_id for e in splits["train"]] == ["1"]
    assert [e.interaction_id for e in splits["validation"]] == ["25"]


def test_exact_near_benchmark_dedup_retains_earliest_and_counts() -> None:
    text = "SYNTHETIC " + " ".join(f"word{i}" for i in range(60))
    benchmark = "SYNTHETIC amber compass measures north near the observatory gate tonight"
    examples = [
        example(3, text=text + " changed"),
        example(1, text=text, digest="same"),
        example(2, text=text, digest="same"),
        example(4, text=benchmark),
        example(5, text="SYNTHETIC wholly different silver whale floating past islands"),
    ]
    counts = Counter()
    result = curate(examples, [benchmark], 0.8, counts)
    assert [e.interaction_id for e in result] == ["1", "5"]
    assert counts == {"exact_duplicate": 1, "near_duplicate": 1, "benchmark_contamination": 1}
    assert shingles("ＳＹＮＴＨＥＴＩＣ Hello!") == shingles("synthetic hello")
    assert not similar(shingles("one"), shingles("two"), 0.8)
    assert not similar(shingles(text), shingles(text), 1.0)  # strictly above threshold


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": ".."},
        {"dataset_id": "../escape"},
        {"tenant_ids": [".."]},
        {"tenant_ids": ["synthetic-a", "synthetic-a"]},
        {"grouping_keys": ["document_family"]},
        {"target_preference_order": ["correction", "correction"]},
        {"near_duplicate_threshold": -1},
    ],
)
def test_dataset_contract_rejects_invalid_specs(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DatasetSpecification.model_validate({**spec().model_dump(), **changes})


def test_source_window_requires_aware_ordered_times() -> None:
    with pytest.raises(ValidationError):
        SourceWindow(start=AT.replace(tzinfo=None), end=AT)
    with pytest.raises(ValidationError):
        SourceWindow(start=AT, end=AT)


def test_eligibility_and_construction_fail_closed(
    dataset_seed: "DatasetSeed",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = dataset_seed
    builder = seed.app.state.datasets
    metadata = seed.app.state.metadata
    at = builder.clock()
    selection = select(metadata, seed.policy, seed.specification, at)
    assert all(e.interaction.tenant_id == "synthetic-a" for e in selection.examples)
    rag = next(e for e in selection.examples if e.retrieval)
    constructor = builder.constructor
    assert constructor.build(rag, seed.specification, at).families == ("refund-policy",)
    evidence = next(c for c in rag.retrieval.candidates if c.supplied_to_model)
    resolver = constructor.sources
    for interaction, retrieval, chunk in [
        (rag.interaction.model_copy(update={"tenant_id": "synthetic-b"}), rag.retrieval, evidence),
        (rag.interaction, rag.retrieval.model_copy(update={"index_version": "absent"}), evidence),
        (rag.interaction, rag.retrieval, evidence.model_copy(update={"content_hash": "wrong"})),
        (rag.interaction, rag.retrieval, evidence.model_copy(update={"licence_class": "unknown"})),
    ]:
        assert resolver.resolve(interaction, retrieval, chunk, "local") is None
    assert resolver.resolve(rag.interaction, rag.retrieval, evidence, "eu-west") is None
    monkeypatch.setattr(resolver, "resolve", lambda *args: None)
    with pytest.raises(Excluded, match="source_unavailable"):
        constructor.build(rag, seed.specification, at)
    plain = next(e for e in selection.examples if not e.retrieval)
    broken = replace(
        plain,
        interaction=plain.interaction.model_copy(
            update={"input": plain.interaction.input.model_copy(update={"content_hash": "wrong"})}
        ),
    )
    with pytest.raises(Excluded, match="payload_hash_mismatch"):
        constructor.build(broken, seed.specification, at)
    seed.policy.training["synthetic-a"] = False
    assert not select(metadata, seed.policy, seed.specification, at).examples


def test_exclusion_reasons_for_failed_redacted_expired_licence_and_feedback(
    dataset_seed: "DatasetSeed",
) -> None:
    seed = dataset_seed
    db = seed.app.state.database
    metadata = seed.app.state.metadata

    def update(index: int, changes: dict[str, object]) -> None:
        item = metadata.get("synthetic-a", Interaction, seed.ids[index]).model_copy(update=changes)
        db.connection.execute(
            "UPDATE interactions SET data=? WHERE tenant_id=? AND record_id=?",
            (item.model_dump_json(), "synthetic-a", item.interaction_id),
        )

    with db.transaction():
        update(2, {"status": "failed"})
        update(3, {"error_code": "persistence_redaction_failed"})
        update(4, {"feedback_ids": ["absent"]})
        db.connection.execute(
            "UPDATE interactions SET expires_at=? WHERE record_id=?",
            ((AT - timedelta(days=1)).isoformat(), seed.ids[5]),
        )
        row = db.connection.execute(
            "SELECT record_id, data FROM retrieval_runs WHERE interaction_id=?", (seed.ids[31],)
        ).fetchone()
        import json

        data = json.loads(row["data"])
        data["candidates"][0]["licence_class"] = "unknown"
        db.connection.execute(
            "UPDATE retrieval_runs SET data=? WHERE record_id=?",
            (json.dumps(data), row["record_id"]),
        )
    counts = select(
        metadata, seed.policy, seed.specification, seed.app.state.datasets.clock()
    ).exclusions
    for reason in [
        "not_completed",
        "persistence_redaction_failed",
        "missing_feedback",
        "expired",
        "unsupported_licence",
    ]:
        assert counts[reason] >= 1


def test_code_revision_unknown_when_git_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic unavailable")

    monkeypatch.setattr("adaptive_llm.datasets.builder.subprocess.run", missing)
    assert code_revision() == "unknown"


def test_golden_fixture_has_twenty_two_synthetic_items(settings: Settings) -> None:
    golden = golden_texts(settings.golden_dir)
    assert len(golden) == 22
    assert all(text.startswith("SYNTHETIC") for text in golden)


def test_source_removal_is_visible_on_next_build(
    dataset_seed: "DatasetSeed",
    tmp_path: Path,
    settings: Settings,
) -> None:
    seed = dataset_seed
    candidate = next(
        e
        for e in select(
            seed.app.state.metadata,
            seed.policy,
            seed.specification,
            seed.app.state.datasets.clock(),
        ).examples
        if e.retrieval
    )
    assert candidate.retrieval is not None
    evidence = next(c for c in candidate.retrieval.candidates if c.supplied_to_model)
    path = tmp_path / "versioned-sources.json"
    resolver = LocalSourceResolver(path)
    assert resolver.resolve(candidate.interaction, candidate.retrieval, evidence, "local") is None
    path.write_text(settings.documents_path.read_text())
    assert (
        resolver.resolve(candidate.interaction, candidate.retrieval, evidence, "local") is not None
    )
    path.write_text("[]")
    assert resolver.resolve(candidate.interaction, candidate.retrieval, evidence, "local") is None


def test_correction_consent_target_order_and_later_negative(dataset_seed: "DatasetSeed") -> None:
    seed = dataset_seed
    builder = seed.app.state.datasets
    headers = {"Authorization": "Bearer synthetic-key-a"}
    iid = seed.ids[2]  # Positive resolution feedback, with no negative signal.

    def candidate() -> Example:
        return next(
            e
            for e in select(
                seed.app.state.metadata, seed.policy, seed.specification, builder.clock()
            ).examples
            if e.interaction.interaction_id == iid
        )

    for consent in [False, True]:
        assert (
            seed.client.post(
                f"/v1/interactions/{iid}/correction",
                headers=headers,
                json={
                    "correction": "SYNTHETIC human corrected target",
                    "training_authorised": consent,
                },
            ).status_code
            == 200
        )
        built = builder.constructor.build(candidate(), seed.specification, builder.clock())
        assert built.target_source == ("correction" if consent else "positive_resolution")
        if consent:
            assert built.row["target"] == "SYNTHETIC human corrected target"
    reordered = seed.specification.model_copy(
        update={
            "target_preference_order": ["production_output", "correction", "positive_resolution"]
        }
    )
    assert (
        builder.constructor.build(candidate(), reordered, builder.clock()).target_source
        == "production_output"
    )
    # A negative posted after the correction reopens exclusion; exactly half is neutral.
    for score in [2, 1]:
        assert (
            seed.client.post(
                f"/v1/interactions/{iid}/feedback",
                headers=headers,
                json={
                    "label_type": "rubric",
                    "value": {"score": score, "max_score": 4},
                },
            ).status_code
            == 200
        )
        selected = select(seed.app.state.metadata, seed.policy, seed.specification, builder.clock())
        assert (iid in {e.interaction.interaction_id for e in selected.examples}) == (score == 2)


@pytest.mark.parametrize("threshold", [0.0, 0.4, 0.8, 1.0])
def test_indexed_near_dedup_matches_exhaustive_comparison(threshold: float) -> None:
    rng = random.Random(19)
    texts = [
        " ".join(rng.choices(["synthetic", "amber", "blue", "green", "red"], k=15))
        for _ in range(60)
    ]
    texts.extend([texts[0], texts[1] + " synthetic", "", "!", "synthetic short"])
    examples = [replace(example(index), text=text) for index, text in enumerate(texts)]
    golden = [texts[4]]
    expected: list[BuiltExample] = []
    expected_counts: Counter[str] = Counter()
    for item in examples:
        if any(item.exact_hash == prior.exact_hash for prior in expected):
            expected_counts["exact_duplicate"] += 1
        elif any(
            similar(shingles(item.text), shingles(prior.text), threshold) for prior in expected
        ):
            expected_counts["near_duplicate"] += 1
        elif similar(shingles(item.text), shingles(golden[0]), threshold):
            expected_counts["benchmark_contamination"] += 1
        else:
            expected.append(item)
    counts: Counter[str] = Counter()
    assert curate(examples, golden, threshold, counts) == expected
    assert counts == expected_counts


def test_near_dedup_skips_pairs_without_shared_shingles(monkeypatch: pytest.MonkeyPatch) -> None:
    comparisons = 0

    def compare(
        left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]], threshold: float
    ) -> bool:
        nonlocal comparisons
        comparisons += 1
        return similar(left, right, threshold)

    monkeypatch.setattr("adaptive_llm.datasets.curation.similar", compare)
    examples = [
        example(index, text=" ".join(f"synthetic{index}word{i}" for i in range(10)))
        for index in range(100)
    ]
    assert curate(examples, ["SYNTHETIC golden benchmark"], 0.8, Counter()) == examples
    assert comparisons == 100  # Golden comparisons only, no 4,950 disjoint example pairs.
