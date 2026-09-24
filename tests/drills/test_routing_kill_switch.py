from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import InferenceRequest, RoutePolicy, uid

if TYPE_CHECKING:
    from conftest import EvaluationSeed


@pytest.mark.drill
def test_kill_switch_under_load_across_process_connections(
    shadow_seed: tuple["EvaluationSeed", str],
    inference_request: InferenceRequest,
) -> None:
    seed, version = shadow_seed
    operator = {"Authorization": "Bearer synthetic-operator-key"}
    user = {"Authorization": "Bearer synthetic-key-a"}
    policy = RoutePolicy(
        eligible_specialist_versions=[version],
        shadow_enabled=True,
        tenant_enabled={"synthetic-a": True},
        task_enabled={"question_answering": True},
    )
    assert (
        seed.client.post(
            "/v1/route-policies",
            headers=operator,
            json=policy.model_dump(mode="json"),
        ).status_code
        == 200
    )
    assert (
        seed.client.post(
            f"/v1/route-policies/{policy.policy_id}/activate",
            headers=operator,
            json={"reason": "SYNTHETIC kill drill activation"},
        ).status_code
        == 200
    )
    app = create_app(Settings(data_dir=seed.directory, outbox_dispatch_enabled=False))
    with TestClient(app) as client:

        def request(index: int) -> int:
            return client.post(
                "/v1/inference",
                headers=user,
                json=inference_request.model_copy(update={"request_id": uid()}).model_dump(
                    mode="json"
                ),
            ).status_code

        assert request(0) == 200
        assert client.portal is not None
        client.portal.call(app.state.shadow.queue.join)
        assert app.state.metrics.get("shadow_completed") == 1
        # Begin a new process-cache interval immediately before the command.
        sleep(1.01)
        assert not client.get("/healthz").json()["kill_switch"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            load = pool.submit(lambda: list(pool.map(request, range(100))))
            started = monotonic()
            assert (
                seed.client.post(
                    "/v1/route-policies/kill-switch",
                    headers=operator,
                    json={"reason": "SYNTHETIC measured emergency disablement"},
                ).status_code
                == 200
            )
            while not client.get("/healthz").json()["kill_switch"]:
                assert monotonic() - started < 2
                sleep(0.01)
            elapsed = monotonic() - started
            assert elapsed < 2
            assert load.result() == [200] * 100
        client.portal.call(app.state.shadow.queue.join)
        completed = app.state.metrics.get("shadow_completed")
        for i in range(10):
            assert request(i) == 200
        client.portal.call(app.state.shadow.queue.join)
        assert app.state.metrics.get("shadow_completed") == completed
        assert app.state.metrics.get("shadow_drops", reason="disabled") >= 10
        print(
            f"kill-switch: 100 concurrent-load requests successful; effect_seconds={elapsed:.3f}; "
            "post-effect_shadow_starts=0; second_instance=honoured"
        )
