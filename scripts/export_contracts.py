"""Export deterministic schema artifacts; run from the repository root."""

import json
from pathlib import Path

from adaptive_llm import contracts
from adaptive_llm.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def export() -> None:
    for name, cls in vars(contracts).items():
        if (
            isinstance(cls, type)
            and issubclass(cls, contracts.Contract)
            and cls is not contracts.Contract
        ):
            folder = "events" if cls is contracts.Event else "schemas"
            path = ROOT / "contracts" / folder / f"{name}.json"
            path.write_text(json.dumps(cls.model_json_schema(), indent=2, sort_keys=True) + "\n")
    (ROOT / "contracts/openapi/openapi.json").write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    export()
