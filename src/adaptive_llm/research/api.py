"""Research-only HTTP wiring; never imported by a disabled app."""

import asyncio
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Response

from adaptive_llm.contracts import EvaluationReport, ModelManifest
from adaptive_llm.gateway.identity import GatewayError, Identity
from adaptive_llm.research.models import (
    ActivationSpecification,
    BaselineSpecification,
    PruneSpecification,
    StudySummary,
)
from adaptive_llm.research.service import ResearchService


def mount(application: FastAPI, authenticate: Callable[..., Identity]) -> None:
    @application.post("/v1/research/baselines", response_model=EvaluationReport)
    async def baseline(
        body: BaselineSpecification, identity: Annotated[Identity, Depends(authenticate)]
    ) -> EvaluationReport:
        service: ResearchService = application.state.research
        try:
            return await asyncio.to_thread(service.baseline, body, identity)
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(503, "research_failed") from None

    @application.post("/v1/research/jobs", response_model=StudySummary | ModelManifest)
    async def job(
        body: ActivationSpecification | PruneSpecification,
        identity: Annotated[Identity, Depends(authenticate)],
    ) -> StudySummary | ModelManifest:
        service: ResearchService = application.state.research
        try:
            if isinstance(body, ActivationSpecification):
                return await asyncio.to_thread(service.study, body, identity)
            return await asyncio.to_thread(service.structured_prune, body, identity)
        except GatewayError:
            raise
        except Exception:
            raise GatewayError(503, "research_failed") from None

    @application.get("/v1/research/studies/{study_id}", response_model=StudySummary)
    async def study(
        study_id: str, identity: Annotated[Identity, Depends(authenticate)]
    ) -> StudySummary:
        service: ResearchService = application.state.research
        return await asyncio.to_thread(service.get, study_id, identity)

    @application.get("/v1/research/studies/{study_id}/aggregates")
    async def aggregates(
        study_id: str, identity: Annotated[Identity, Depends(authenticate)]
    ) -> Response:
        service: ResearchService = application.state.research
        _, raw = await asyncio.to_thread(service.store.get, study_id, identity)
        return Response(raw, media_type="application/octet-stream")
