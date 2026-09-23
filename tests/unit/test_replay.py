import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import replace

import pytest

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import InferenceRequest, PolicyDecision, now
from adaptive_llm.events import InMemoryEventSink
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.gateway.service import InferenceService
from adaptive_llm.providers import FakeProvider, ProviderRequest, ProviderResult

ServiceFactory = Callable[[Settings], Awaitable[InferenceService]]


@pytest.fixture
async def service_factory() -> AsyncIterator[ServiceFactory]:
    async with AsyncExitStack() as stack:

        async def start(settings: Settings) -> InferenceService:
            app = create_app(settings)
            await stack.enter_async_context(app.router.lifespan_context(app))
            return app.state.inference

        yield start


async def test_replay_capacity_evicts_oldest_completed_entry(
    inference_request: InferenceRequest, identity: Identity, service_factory: ServiceFactory
) -> None:
    assert Settings().replay_capacity == 10_000
    service = await service_factory(Settings(replay_capacity=2))
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
    assert not service._replays


async def test_sql_replay_cap_is_tenant_scoped_and_removes_evicted_payloads(
    inference_request: InferenceRequest,
    identity: Identity,
    service_factory: ServiceFactory,
) -> None:
    service = await service_factory(Settings(replay_capacity=2))
    meta, payloads = service.persistence.metadata, service.persistence.payloads
    tenant_b = replace(identity, tenant_id="synthetic-b")
    first = await service.infer(inference_request, identity)
    saved = meta.get_replay(
        identity.tenant_id, inference_request.application_id, inference_request.request_id, now()
    )
    assert saved is not None
    other_tenant = await service.infer(inference_request, tenant_b)
    for index in range(2):
        await service.infer(
            inference_request.model_copy(update={"request_id": f"synthetic-{index}"}), identity
        )
    assert (
        meta.get_replay(
            identity.tenant_id,
            inference_request.application_id,
            inference_request.request_id,
            now(),
        )
        is None
    )
    assert payloads.get(identity.tenant_id, saved.response_ref, now()) is None
    assert (
        await service.infer(inference_request, tenant_b)
    ).interaction_id == other_tenant.interaction_id
    assert (await service.infer(inference_request, identity)).interaction_id != first.interaction_id
    assert not service._replays


@pytest.mark.parametrize("capacity", [0, -1])
def test_replay_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError, match="^invalid_replay_capacity$"):
        Settings(replay_capacity=capacity)


@pytest.mark.parametrize("status", [500, 502, 503, 504, None])
async def test_failed_reservations_are_released(
    inference_request: InferenceRequest,
    identity: Identity,
    status: int | None,
    service_factory: ServiceFactory,
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
    service = await service_factory(Settings(policy=policy, replay_capacity=1))
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
    inference_request: InferenceRequest, identity: Identity, service_factory: ServiceFactory
) -> None:
    provider, sink = FirstCallWaits(), InMemoryEventSink()
    service = await service_factory(Settings(provider=provider, events=sink))
    task = asyncio.create_task(service.infer(inference_request, identity))
    await asyncio.wait_for(provider.entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not service._replays
    result = await service.infer(inference_request, identity)
    assert not result.replayed
    async with asyncio.timeout(2):
        while len(sink.events) < 10:  # noqa: ASYNC110 - observe asynchronous delivery
            await asyncio.sleep(0.01)
    assert len(sink.events) == 10
    assert sink.events[3].data.finish_reason == "cancelled"
    assert sink.events[0].data.interaction_id != result.interaction_id
    assert sink.events[-1].data.status == "completed"
    assert provider.calls == 2


async def test_completed_entries_do_not_count_towards_in_flight_capacity(
    inference_request: InferenceRequest, identity: Identity, service_factory: ServiceFactory
) -> None:
    provider = FirstCallWaits()
    service = await service_factory(Settings(provider=provider, replay_capacity=2))
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
        replayed = await service.infer(second_request, identity)
        assert replayed.interaction_id == second.interaction_id
        assert replayed.replayed
        assert len(service._replays) == 1
    finally:
        provider.release.set()
        await task


async def test_full_active_cache_rejects_new_work_without_losing_reservation(
    inference_request: InferenceRequest, identity: Identity, service_factory: ServiceFactory
) -> None:
    provider = FirstCallWaits()
    service = await service_factory(Settings(provider=provider, replay_capacity=1))
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
