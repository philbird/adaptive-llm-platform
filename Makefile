.PHONY: ci dev check integration contracts sbom migrate retention-sweep dispatch-once dead-letters redeliver rotate-key backup restore drills dataset-build evaluate

DATA_DIR ?= .local
ENVIRONMENT ?= local
TENANT ?= synthetic-a
KEYRING ?= $(PAYLOAD_KEYRING)
STORAGE = uv run --locked python -m adaptive_llm.storage
TRAINING = uv run --locked python -m adaptive_llm.training
TRAINING_ARGS = --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)"
BACKEND ?= fake
# Pass notes via environment to avoid expanding user text as shell code.
export NOTE
export TENANT APP
.PHONY: issue-key record-openrouter
.PHONY: train promote rollback models
.PHONY: check-without-training

# The default verification gate must not resolve or download packages from the network.
check integration check-without-training contracts: export UV_OFFLINE = 1

train:
	$(TRAINING) train $(TRAINING_ARGS) --backend "$(BACKEND)" --spec "$(SPEC)" $(if $(POLICY),--policy "$(POLICY)",)

promote:
	$(TRAINING) promote $(TRAINING_ARGS) --model "$(MODEL)" --to "$(TO)" --note "$$NOTE" $(if $(EVALUATION_ID),--evaluation-id "$(EVALUATION_ID)",)

rollback:
	$(TRAINING) rollback $(TRAINING_ARGS) --deployment "$(DEPLOYMENT)" --note "$$NOTE"

models:
	$(TRAINING) models $(TRAINING_ARGS)

ci: check integration sbom
	uv run --locked pytest tests/security tests/load -s
	uv run --locked pytest tests/drills -m drill -s
	uv run --locked pytest -m smoke -s
	$(MAKE) check-without-training
	@echo "local ci: all gates passed"

check-without-training:
	uv run --locked python scripts/check_without_training.py tests -q

dev:
	uv run --locked uvicorn adaptive_llm.app:app --host 127.0.0.1 --port 8000 --reload --no-access-log

dataset-build:
	uv run --locked python -m adaptive_llm.datasets --spec "$(SPEC)" --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" $(if $(POLICY),--policy "$(POLICY)",)

evaluate:
	uv run --locked python -m adaptive_llm.evaluation --spec "$(SPEC)" --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)"

issue-key:
	uv run --locked python -m adaptive_llm.gateway.identity --tenant "$$TENANT" --app "$$APP"

record-openrouter:
	uv run --locked python -m adaptive_llm.providers.cassette

migrate:
	uv run --locked python -m adaptive_llm.storage migrate --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)"

retention-sweep:
	uv run --locked python -m adaptive_llm.storage retention-sweep --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --tenant "$(TENANT)"

dispatch-once:
	$(STORAGE) dispatch-once --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)"

dead-letters:
	$(STORAGE) dead-letters --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --tenant "$(TENANT)"

redeliver:
	$(STORAGE) redeliver --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --event-id "$(EVENT_ID)"

rotate-key:
	$(STORAGE) rotate-key --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --tenant "$(TENANT)" --new-key-version "$(NEW_KEY_VERSION)" --keyring "$(KEYRING)"

backup:
	$(STORAGE) backup --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --out "$(OUT)"

restore:
	$(STORAGE) restore --data-dir "$(DATA_DIR)" --environment "$(ENVIRONMENT)" --out "$(OUT)" $(if $(filter 1,$(FORCE)),--force,)

drills:
	uv run --locked pytest tests/drills -m drill -s

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

.PHONY: sign-rotate
sign-rotate:
	uv run --locked python -m adaptive_llm.signing --private-key "$(SIGNING_KEY_PATH)" --public-keys "$(SIGNING_PUBLIC_KEYS)" --key-id "$(SIGNING_KEY_ID)"
