"""Injectable CPU trainer boundary; real LoRA belongs to slice 3b."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from adaptive_llm.contracts import DatasetManifest, ResourceUsage, TrainingJobSpecification


class Trainer(Protocol):
    @property
    def architecture(self) -> str: ...

    def train(
        self,
        specification: TrainingJobSpecification,
        dataset: DatasetManifest,
        directory: Path,
        checkpoint_refs: list[str],
        checkpoint: Callable[[str], None],
    ) -> ResourceUsage: ...
