import asyncio

import pytest

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import InferenceRequest, PolicyDecision
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.gateway.service import InferenceService
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult


async def test_replay_capacity_evicts_oldest_completed_entry(
    inference_request: InferenceRequest, identity: Identity
) -> None:
    assert Settings().replay_capacity == 10_000
    service: InferenceService = create_app(Settings(replay_capacity=2)).state.inference
    second_request = inference_request.model_copy(update={"request_id": "synthetic-second"})
    third_request = inference_request.model_copy(update={"request_id": "synthetic-third"})
    first = await service.infer(inference_request, identity)
    second = await service.infer(second_request, identity)
    assert (await service.infer(inference_request, identity)).replayed
    third = await service.infer(third_request, identity)
    assert (await service.infer(second_request, identity)).interaction_id == second.interaction_id
    assert (await service.infer(third_request, identity)).interaction_id == third.interaction_id
    evicted = await service.infer(inference_request, identity)
    assert not evicted.replayed
    assert evicted.interaction_id != first.interaction_id
    assert len(service._replays) == 2


@pytest.mark.parametrize("capacity", [0, -1])
def test_replay_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError, match="^invalid_replay_capacity$"):
        Settings(replay_capacity=capacity)


@pytest.mark.parametrize("status", [500, 502, 503, 504, None])
async def test_failed_reservations_are_released(
    inference_request: InferenceRequest, identity: Identity, status: int | None
) -> None:
    class FailsOnce:
        calls = 0

        def decide(self, identity: Identity, application_id: str) -> PolicyDecision:
            self.calls += 1
            if self.calls == 1:
                if status is None:
                    raise RuntimeError("synthetic-internal-failure")
                raise GatewayError(status, "synthetic-transient-failure")
            return PolicyDecision(policy_version="synthetic-1", retention_seconds=1)

    policy = FailsOnce()
    service: InferenceService = create_app(
        Settings(policy=policy, replay_capacity=1)
    ).state.inference
    with pytest.raises(GatewayError) as failure:
        await service.infer(inference_request, identity)
    assert failure.value.status_code == (status or 500)
    assert not service._replays
    result = await service.infer(inference_request, identity)
    assert not result.replayed
    assert (await service.infer(inference_request, identity)).replayed
    assert policy.calls == 2


class FirstCallWaits(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            await self.release.wait()
        return await super().generate(request)


async def test_cancellation_releases_reservation_and_can_retry(
    inference_request: InferenceRequest, identity: Identity
) -> None:
    provider, sink = FirstCallWaits(), InMemoryEventSink()
    service: InferenceService = create_app(Settings(provider=provider, events=sink)).state.inference
    task = asyncio.create_task(service.infer(inference_request, identity))
    await asyncio.wait_for(provider.entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not service._replays
    result = await service.infer(inference_request, identity)
    assert not result.replayed
    assert len(sink.events) == 10
    assert sink.events[3].data.finish_reason == "cancelled"
    assert sink.events[0].data.interaction_id != result.interaction_id
    assert sink.events[-1].data.status == "completed"
    assert provider.calls == 2


async def test_capacity_evicts_completed_entry_without_evicting_active_reservation(
    inference_request: InferenceRequest, identity: Identity
) -> None:
    provider = FirstCallWaits()
    service: InferenceService = create_app(
        Settings(provider=provider, replay_capacity=2)
    ).state.inference
    task = asyncio.create_task(service.infer(inference_request, identity))
    await asyncio.wait_for(provider.entered.wait(), timeout=2)
    try:
        second_request = inference_request.model_copy(update={"request_id": "synthetic-second"})
        second = await service.infer(second_request, identity)
        await service.infer(
            inference_request.model_copy(update={"request_id": "synthetic-third"}), identity
        )
        with pytest.raises(GatewayError, match="^request_in_progress$") as duplicate:
            await service.infer(inference_request, identity)
        assert duplicate.value.status_code == 409
        evicted = await service.infer(second_request, identity)
        assert evicted.interaction_id != second.interaction_id
        assert len(service._replays) == 2
    finally:
        provider.release.set()
        await task


async def test_full_active_cache_rejects_new_work_without_losing_reservation(
    inference_request: InferenceRequest, identity: Identity
) -> None:
    provider = FirstCallWaits()
    service: InferenceService = create_app(
        Settings(provider=provider, replay_capacity=1)
    ).state.inference
    task = asyncio.create_task(service.infer(inference_request, identity))
    await asyncio.wait_for(provider.entered.wait(), timeout=2)
    try:
        with pytest.raises(GatewayError, match="^replay_capacity_exceeded$") as failure:
            await service.infer(
                inference_request.model_copy(update={"request_id": "synthetic-new"}), identity
            )
        assert failure.value.status_code == 429
        with pytest.raises(GatewayError, match="^request_in_progress$"):
            await service.infer(inference_request, identity)
        assert provider.calls == 1
        assert len(service._replays) == 1
    finally:
        provider.release.set()
        result = await task
    assert (
        await service.infer(inference_request, identity)
    ).interaction_id == result.interaction_id
