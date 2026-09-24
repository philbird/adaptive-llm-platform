"""Deterministic exact/near deduplication, benchmark filtering and joint grouping."""

import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from adaptive_llm.contracts import DatasetSpecification, Split
from adaptive_llm.datasets.construction import BuiltExample

SPLITS: tuple[Split, ...] = ("train", "validation", "test")


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    size = min(5, len(words))
    return frozenset(tuple(words[i : i + size]) for i in range(len(words) - size + 1))


def similar(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]], threshold: float
) -> bool:
    union = left | right
    return (len(left & right) / len(union) if union else 1.0) > threshold


def golden_texts(directory: Path) -> list[str]:
    texts: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.name == "distillation-training.jsonl":
            continue  # Explicit training partition; never used by the evaluation suites.
        for line in path.read_text().splitlines():
            row = json.loads(line)
            texts.append(row["input"] + "\n" + row["target"])
    if not texts:
        raise ValueError("golden_set_required")
    return texts


def curate(
    examples: list[BuiltExample], golden: list[str], threshold: float, exclusions: Counter[str]
) -> list[BuiltExample]:
    benchmarks = [shingles(text) for text in golden]
    seen: set[str] = set()
    accepted: list[BuiltExample] = []
    grams: list[frozenset[tuple[str, ...]]] = []
    postings: dict[tuple[str, ...], set[int]] = {}
    for example in sorted(examples, key=lambda e: (e.started_at, e.interaction_id)):
        current = shingles(example.text)
        overlapping: set[int] = set()
        for shingle in current:
            overlapping.update(postings.get(shingle, ()))
        if example.exact_hash in seen:
            exclusions["exact_duplicate"] += 1
        elif any(similar(current, grams[index], threshold) for index in sorted(overlapping)):
            exclusions["near_duplicate"] += 1
        elif any(similar(current, item, threshold) for item in benchmarks):
            exclusions["benchmark_contamination"] += 1
        else:
            seen.add(example.exact_hash)
            for shingle in current:
                postings.setdefault(shingle, set()).add(len(grams))
            grams.append(current)
            accepted.append(example)
    return accepted


def split_examples(
    examples: list[BuiltExample], spec: DatasetSpecification
) -> dict[Split, list[BuiltExample]]:
    """Components span BOTH grouping keys; missing subjects never form a shared group."""
    ordered = sorted(examples, key=lambda e: (e.started_at, e.interaction_id))
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, example in enumerate(ordered):
        keys = [("family", family) for family in example.families]
        if example.subject is not None:
            keys.append(("subject", example.subject))
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    groups: dict[int, list[BuiltExample]] = {}
    for index, example in enumerate(ordered):
        groups.setdefault(root(index), []).append(example)
    components = list(groups.values())
    rng = random.Random(spec.seed)
    rng.shuffle(components)
    result: dict[Split, list[BuiltExample]] = {split: [] for split in SPLITS}
    assigned = 0
    for group in components:
        split: Split
        if spec.time_split:
            # A component crossing a cutoff moves in full to its latest partition.
            latest = max(e.started_at for e in group)
            split = (
                "train"
                if latest < spec.time_split.train_end
                else "validation"
                if latest < spec.time_split.validation_end
                else "test"
            )
        else:
            fraction = assigned / max(1, len(ordered))
            split = "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"
        result[split].extend(group)
        assigned += len(group)
    for group in result.values():
        group.sort(key=lambda e: (e.started_at, e.interaction_id))
    return result
