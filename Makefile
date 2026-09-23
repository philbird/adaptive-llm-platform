.PHONY: dev check integration contracts sbom migrate retention-sweep

DATA_DIR ?= .local
ENVIRONMENT ?= local
TENANT ?= synthetic-a

dev:
	uv run --locked uvicorn adaptive_llm.app:app --host 127.0.0.1 --port 8000 --reload --no-access-log

migrate:
	uv run --locked python -m adaptive_llm.storage migrate --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)"

retention-sweep:
	uv run --locked python -m adaptive_llm.storage retention-sweep --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --tenant "$(TENANT)"

check:
	uv run --locked ruff format --check .
	uv run --locked ruff check .
	uv run --locked mypy src/adaptive_llm
	uv run --locked pytest tests/unit tests/contract

integration:
	uv run --locked pytest tests/integration

contracts:
	uv run --locked python scripts/export_contracts.py

sbom:
	mkdir -p reports
	uv export --locked --no-dev --no-emit-project --format requirements-txt --output-file reports/sbom-requirements.txt
	uv run --locked pip-audit --requirement reports/sbom-requirements.txt --no-deps --disable-pip --format cyclonedx-json --output sbom.json
