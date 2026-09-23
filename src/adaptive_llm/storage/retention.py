"""Synchronous local retention sweep; the caller supplies the tenant and UTC clock."""

from collections.abc import Callable
from datetime import datetime

from adaptive_llm.contracts import now
from adaptive_llm.storage import MetadataStore, PayloadStore


def sweep(
    metadata: MetadataStore,
    payloads: PayloadStore,
    tenant_id: str,
    *,
    clock: Callable[[], datetime] = now,
) -> int:
    at = clock()
    with metadata.transaction():
        expired = metadata.expired(tenant_id, at)
        for interaction in expired:
            metadata.clear_refs(tenant_id, interaction.interaction_id, "expired")
            payloads.delete_interaction(tenant_id, interaction.interaction_id)
        metadata.expire_replays(tenant_id, at)
    return len(expired)
