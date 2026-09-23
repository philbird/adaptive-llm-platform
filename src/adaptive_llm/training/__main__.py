"""Local operator CLI for the same training and lifecycle services as HTTP."""

import argparse
import asyncio
import json
from pathlib import Path

from adaptive_llm.app import Settings, create_app
from adaptive_llm.contracts import PromotionRequest, TrainingJobSpecification
from adaptive_llm.gateway.identity import Authenticator, GatewayError
from adaptive_llm.registry import ModelRegistry
from adaptive_llm.training.service import TrainingOrchestrator


async def run(args: argparse.Namespace) -> None:
    settings = Settings(
        data_dir=args.data_dir,
        environment=args.environment,
        outbox_dispatch_enabled=False,
        **({"policy_path": args.policy} if args.policy else {}),
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        authenticator: Authenticator = app.state.authenticator
        identity = authenticator.authenticate(
            f"Bearer {args.operator_key}", "synthetic-training-operator"
        )
        if identity.environment != settings.environment:
            raise GatewayError(403, "environment_forbidden")
        registry: ModelRegistry = app.state.registry
        if args.command == "train":
            spec = TrainingJobSpecification.model_validate_json(args.spec.read_text())
            trainer: TrainingOrchestrator = app.state.training
            job = await asyncio.to_thread(trainer.run, spec, identity)
            print(job.model_dump_json(indent=2))
        elif args.command == "promote":
            model = await asyncio.to_thread(
                registry.promote,
                PromotionRequest(
                    model_version=args.model,
                    target_state=args.to,
                    reason=args.note,
                    evaluation_id=args.evaluation_id,
                ),
                identity,
            )
            print(model.model_dump_json(indent=2))
        elif args.command == "rollback":
            models = await asyncio.to_thread(
                registry.rollback, args.deployment, identity, args.note
            )
            print(json.dumps([m.model_dump(mode="json") for m in models]))
        else:
            models = await asyncio.to_thread(registry.models, identity)
            print(
                json.dumps(
                    [
                        {
                            "registry_id": m.registry_id,
                            "version": m.version,
                            "state": m.state,
                            "evaluation_ids": list(m.evaluation_reports),
                        }
                        for m in models
                    ]
                )
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "promote", "rollback", "models"])
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    parser.add_argument("--environment", choices=["local"], default="local")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--operator-key", default="synthetic-operator-key")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--to")
    parser.add_argument("--note")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--deployment")
    args = parser.parse_args(argv)
    if args.command == "train" and args.spec is None:
        parser.error("train requires --spec")
    if args.command == "promote" and not (args.model and args.to and args.note):
        parser.error("promote requires --model, --to and --note")
    if args.command == "rollback" and not (args.deployment and args.note):
        parser.error("rollback requires --deployment and --note")
    try:
        asyncio.run(run(args))
    except Exception:
        parser.exit(1, "training_control_failed\n")


if __name__ == "__main__":
    main()
