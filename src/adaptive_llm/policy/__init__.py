"""Local purpose policy and processing-only credential/payment redaction."""

import json
import re
from pathlib import Path
from typing import Protocol

from adaptive_llm.contracts import InferenceRequest, PolicyDecision
from adaptive_llm.gateway.identity import Identity


class PolicyEngine(Protocol):
    def decide(self, identity: Identity, application_id: str) -> PolicyDecision: ...


class LocalPolicyEngine:
    def __init__(self, path: Path) -> None:
        config = json.loads(path.read_text())
        self._version: str = config["policy_version"]
        self._tenants = {
            tenant: PolicyDecision.model_validate({"policy_version": self._version, **decision})
            for tenant, decision in config["tenants"].items()
        }

    def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
        decision = self._tenants.get(identity.tenant_id)
        if decision is None or application_id not in identity.application_ids:
            return PolicyDecision(
                policy_version=self._version, processing_allowed=False, retention_seconds=3600
            )
        return decision.model_copy(deep=True)


class ProcessingRedactor:
    version = "processing-regex-local-1"
    _patterns = (
        ("credentials", re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")),
        (
            "secrets",
            re.compile(
                r"(?i)\b(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*"
                r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
            ),
        ),
        ("api_keys", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")),
    )
    _cards = re.compile(r"(?<![\w-])(?:[0-9][ -]?){12,18}[0-9](?![\w-])")

    @staticmethod
    def _is_card(value: str) -> bool:
        # A checksum avoids treating every long numeric business identifier as a card.
        digits = [int(char) for char in value if char.isdigit()]
        total = 0
        for position, digit in enumerate(reversed(digits)):
            if position % 2:
                digit *= 2
                digit = digit - 9 if digit > 9 else digit
            total += digit
        return bool(total) and total % 10 == 0

    def redact_text(self, content: str) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = {}
        for name, pattern in self._patterns:
            content, count = pattern.subn("[REDACTED]", content)
            if count:
                counts[name] = counts.get(name, 0) + count

        def redact_card(match: re.Match[str]) -> str:
            if not self._is_card(match.group()):
                return match.group()
            counts["card_numbers"] = counts.get("card_numbers", 0) + 1
            return "[REDACTED]"

        return self._cards.sub(redact_card, content), counts

    def redact(self, request: InferenceRequest) -> tuple[InferenceRequest, dict[str, int]]:
        counts: dict[str, int] = {}

        def redact_text(content: str) -> str:
            redacted, found = self.redact_text(content)
            for name, count in found.items():
                counts[name] = counts.get(name, 0) + count
            return redacted

        messages = [
            message.model_copy(update={"content": redact_text(message.content)})
            for message in request.messages
        ]
        metadata = {key: redact_text(value) for key, value in request.metadata.items()}
        return request.model_copy(update={"messages": messages, "metadata": metadata}), counts
