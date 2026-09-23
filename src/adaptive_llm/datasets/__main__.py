"""Build with the same injected services and operator authorization as HTTP."""

import argparse
import asyncio
from pathlib import Path

from adaptive_llm.app import ROOT, Settings, create_app
from adaptive_llm.contracts import DatasetSpecification, Environment
from adaptive_llm.datasets.builder import DatasetBuilder
from adaptive_llm.gateway.identity import Authenticator, GatewayError


async def run(
    spec: Path,
    data_dir: Path,
    environment: Environment,
    operator_key: str,
    policy_path: Path = ROOT / "configs/policy/local.json",
) -> None:
    specification = DatasetSpecification.model_validate_json(
        await asyncio.to_thread(spec.read_text)
    )
    app = create_app(
        Settings(
            data_dir=data_dir,
            environment=environment,
            outbox_dispatch_enabled=False,
            policy_path=policy_path,
        )
    )
    async with app.router.lifespan_context(app):
        authenticator: Authenticator = app.state.authenticator
        identity = authenticator.authenticate(f"Bearer {operator_key}", None)
        if identity.environment != environment:
            raise GatewayError(403, "environment_forbidden")
        builder: DatasetBuilder = app.state.datasets
        manifest = await asyncio.to_thread(builder.build, specification, identity)
        print(manifest.model_dump_json(indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build an immutable local synthetic dataset")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(".local"))
    parser.add_argument(
        "--environment", choices=["local", "development", "staging", "production"], default="local"
    )
    parser.add_argument("--operator-key", default="synthetic-operator-key")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/policy/local.json")
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args.spec, args.data_dir, args.environment, args.operator_key, args.policy))
    except Exception:
        # Never stringify validation errors or payload/provider exceptions.
        parser.exit(1, "dataset_build_failed\n")


if __name__ == "__main__":
    main()
