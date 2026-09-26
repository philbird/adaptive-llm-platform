"""Run the same operator-authorized evaluator from a local CLI."""

import argparse
import asyncio
import os
from dataclasses import replace
from pathlib import Path

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import Environment, EvaluationInput
from adaptive_llm.evaluation.service import Evaluator
from adaptive_llm.gateway.identity import Authenticator, GatewayError
from adaptive_llm.providers.cassette import CassetteProvider
from adaptive_llm.routing import FoundationRouter


async def run(spec: Path, data_dir: Path, environment: Environment, operator_key: str) -> None:
    request = EvaluationInput.model_validate_json(await asyncio.to_thread(spec.read_text))
    settings = Settings(data_dir=data_dir, environment=environment, outbox_dispatch_enabled=False)
    foundation = FoundationRouter(settings.routing_path).deployment
    if foundation.model_provider == "openrouter":
        # The evaluation CLI always replays. Recording is a separate explicit host command.
        settings = replace(settings, provider=CassetteProvider(foundation.model_id))
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        authenticator: Authenticator = app.state.authenticator
        identity = authenticator.authenticate(
            f"Bearer {operator_key}", "synthetic-evaluation-operator"
        )
        if identity.environment != environment:
            raise GatewayError(403, "environment_forbidden")
        evaluator: Evaluator = app.state.evaluations
        report = await asyncio.to_thread(evaluator.evaluate, request, identity)
        print(report.model_dump_json(indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and lock a local synthetic foundation baseline"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    parser.add_argument("--environment", choices=["local"], default="local")
    parser.add_argument(
        "--operator-key", default=os.environ.get("ADAPTIVE_OPERATOR_KEY", "synthetic-operator-key")
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args.spec, args.data_dir, args.environment, args.operator_key))
    except Exception:
        parser.exit(1, "evaluation_failed\n")


if __name__ == "__main__":
    main()
