"""The persistence pass adds PII removal without changing transient model inputs."""

import re
from typing import Protocol

from adaptive_llm.contracts import PolicyDecision
from adaptive_llm.policy import ProcessingRedactor


class PersistenceRedactor(Protocol):
    @property
    def version(self) -> str: ...

    def redact_text(self, content: str, policy: PolicyDecision) -> tuple[str, dict[str, int]]: ...


class LocalPersistenceRedactor:
    version = "persistence-regex-local-1"
    _emails = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-z]{2,}(?![\w.-])")
    _phones = re.compile(r"(?<![\w-])\+?(?:\([0-9]{2,4}\)|[0-9])[0-9 ().-]{5,}[0-9](?!\w)")

    def __init__(self) -> None:
        self._processing = ProcessingRedactor()

    def redact_text(self, content: str, policy: PolicyDecision) -> tuple[str, dict[str, int]]:
        # All local tenants use the same full rule set; policy controls whether refs are kept.
        content, counts = self._processing.redact_text(content)
        content, count = self._emails.subn("[REDACTED]", content)
        if count:
            counts["emails"] = count

        def phone(match: re.Match[str]) -> str:
            if not 7 <= sum(char.isdigit() for char in match.group()) <= 15:
                return match.group()
            counts["phones"] = counts.get("phones", 0) + 1
            return "[REDACTED]"

        return self._phones.sub(phone, content), counts
