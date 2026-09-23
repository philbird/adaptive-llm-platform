.PHONY: dev check integration contracts sbom

dev:
	uv run --locked uvicorn adaptive_llm.app:app --host 127.0.0.1 --port 8000 --reload

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
