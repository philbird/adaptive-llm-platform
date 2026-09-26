"""Bounded JSON schemas with local-only reference resolution and content-free failures."""

import json

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from pydantic import JsonValue
from referencing import Registry


def canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def check_schema(schema: dict[str, JsonValue]) -> None:
    try:
        if len(canonical(schema).encode("utf-8")) > 8192:
            raise ValueError
        Draft202012Validator.check_schema(schema)
        # External references are unsupported at ingress, including references in unused defs.
        pending: list[JsonValue] = [schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in {"$ref", "$dynamicRef"} and (
                        not isinstance(value, str) or not value.startswith("#")
                    ):
                        raise ValueError
                    pending.append(value)
            elif isinstance(item, list):
                pending.extend(item)
    except Exception:
        raise ValueError("invalid_json_schema") from None


def reject_constant(value: str) -> None:
    raise ValueError("invalid_json_constant")


def matches_schema(content: str, schema: dict[str, JsonValue]) -> bool:
    try:
        value = json.loads(content, parse_constant=reject_constant)
        # An explicit empty registry prevents jsonschema's legacy remote retrieval behavior.
        return bool(Draft202012Validator(schema, registry=Registry()).is_valid(value))
    except Exception:
        return False
