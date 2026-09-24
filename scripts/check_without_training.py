"""Run the default suite with optional training imports unavailable (no uninstall needed)."""

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


class WithoutTraining(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname.split(".")[0] in {"torch", "transformers", "peft", "safetensors"}:
            raise ImportError("optional training group intentionally unavailable")
        return None


if __name__ == "__main__":
    sys.meta_path.insert(0, WithoutTraining())
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:] or ["tests", "-q"]))
