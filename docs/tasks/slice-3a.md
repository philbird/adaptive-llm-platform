# Task: slice 3a — model registry, training orchestration and promotion state machine

Milestone 3 begins. Implement the lifecycle around a specialist without any real weights yet, as
described in the specification (`docs/spec/`, sections 7.10, 7.12, 8.7, 9.3 training and
promotion endpoints, 9.4 `training.completed.v1` and `deployment.changed.v1`, 11.3 configuration
surface only, 11.6, 12.3, 20.3, 20.4, 21.2 step 10, 21.4). Read `AGENTS.md`, `docs/status.md`,
`docs/runbooks/dataset-build.md` and `docs/runbooks/evaluation.md` first. Milestones 1 and 2
are on `main`. The real LoRA trainer with PyTorch and PEFT is slice 3b behind an optional
dependency group; this slice makes everything around it real, using a **deterministic fake
trainer** that produces a signed artifact, so the lifecycle, lineage and gates are proven before
the heavy dependency is added.

## Deliverables

1. **Contracts** in `contracts.py` (then `make contracts`): `TrainingJobSpecification`
   (job id, job type limited to `adapter` for now, dataset id and version, base model id and
   revision, tokenizer id, chat template version, adapter config with target modules, rank,
   alpha, dropout, learning rate, precision, gradient accumulation, seed, hardware class,
   container digest or `"local"`, code revision filled by the server), `TrainingJob` (spec,
   state `queued|running|succeeded|failed|cancelled`, started/completed, resource usage,
   checkpoint refs, artifact ref, artifact digest, failure code), `ModelManifest` matching spec
   8.7 (registry id, immutable version, base model and revision and licence, adapter
   architecture, tokenizer and chat template versions, dataset manifests with weights,
   training job id, code revision, container digest, configuration digest, seed, hardware
   class, evaluation report ids with pass/fail, intended tasks, excluded tasks, languages,
   context limit, safety notes, artifact hashes and MAC, storage location, lifecycle history),
   `PromotionRequest` (model version, target state, actor, reason, evaluation id),
   `LifecycleTransition` (from, to, actor, at, reason, evaluation id). All `Record`s. Fill in
   `TrainingCompleted` and `DeploymentChanged` payloads as needed.
2. **Registry** in `src/adaptive_llm/registry/`: a `ModelRegistry` Protocol and SQLite
   implementation in the **control database** introduced by slice 2b
   (`<data_dir>/evaluations/control/<environment>.sqlite3`; rename the directory to
   `<data_dir>/control/` and update the evaluation store, runbooks and tests accordingly), with
   migration 0007: `training_jobs`, `model_versions`, `lifecycle_transitions`, `deployments`.
   Control-plane migrations must live in their own directory `migrations/control/` so the
   tenant database no longer receives control tables and vice versa; split 0006 accordingly
   and keep the runner generic. `make backup`, `make restore` and the backup/restore drill must
   cover both databases atomically (back up tenant first, then control; restore refuses a
   pair whose versions do not match). Lifecycle states exactly as spec 7.12:
   `candidate → evaluating → approved → shadow → canary → production → deprecated`, with
   `revoked` reachable from any state. Transitions are validated by a pure state-machine
   function with a table of allowed edges and required preconditions: `evaluating → approved`
   requires a passed `EvaluationReport` for that model version against the locked baseline of
   the same dataset version; `approved → shadow`, `shadow → canary`, `canary → production`
   require an explicit operator actor and reason and are recorded with them; `revoked` requires
   a reason; nothing may skip `evaluating`. Every transition is written with the actor, time
   and reason and emitted as `deployment.changed.v1` per tenant of the training dataset.
   A state-machine unit test enumerates every pair of states and asserts exactly the allowed
   edges succeed.
3. **Training orchestrator** in `src/adaptive_llm/training/`: `POST /v1/training/jobs`
   (operator key) validates that the dataset version exists, its manifest MAC verifies, its
   `approval.status == "approved"` (add a minimal `POST /v1/datasets/{id}/versions/{version}/approval`
   operator endpoint that records actor and reason and emits nothing new; the manifest
   approval field is updated in the store, never in the immutable shard directory), and that the
   tenant policy currently allows training for every tenant in the manifest. It then runs the
   job off the event loop through a `Trainer` Protocol, persists `TrainingJob` state
   transitions, writes checkpoints and the final artifact under
   `<data_dir>/models/<registry_id>/<version>/`, computes the artifact digest and a MAC with a
   new keyring purpose, registers a `ModelManifest` in state `candidate`, and emits
   `training.completed.v1`. `GET /v1/training/jobs/{job_id}` returns status, lineage and
   metrics. Jobs are resumable: a job interrupted after a checkpoint resumes from it on the
   next run of the same job id (test by injecting a failure after checkpoint 1 and rerunning).
   Never overwrite a model version.
4. **Deterministic fake trainer** in `src/adaptive_llm/training/fake.py`: reads the encrypted
   train shard through the existing cipher, "trains" by computing a deterministic digest over
   the ordered examples and the adapter config, writes `adapter_config.json` and a small
   `adapter_weights.bin` whose bytes are derived from that digest (no real tensors), writes two
   checkpoints with a `training_report.json` (loss curve that is a deterministic function of
   the seed, example count and steps), and returns resource usage. Reproducibility test: two
   runs from the same specification produce byte-identical artifacts and manifests except ids
   and timestamps.
5. **Evaluating a candidate**: extend `EvaluationDeployment` so a registered model version can
   be evaluated by loading its artifact through a `SpecialistProvider` that, for the fake
   adapter, behaves as the fake foundation with the adapter digest appended as a hidden marker
   in `model_version` (so reports can prove which artifact they evaluated). `POST /v1/evaluations`
   with `candidate_deployment_id` equal to a registered model version runs the existing five
   suites against it, and a passed report moves the version `evaluating → approved` only through
   `POST /v1/models/{model_version}/promotion-requests`.
6. **Promotion API**: `POST /v1/models/{model_version}/promotion-requests` (operator key)
   takes a `PromotionRequest`, validates through the state machine, records the transition and
   returns the updated manifest. `POST /v1/deployments/{deployment_id}/rollback` moves the
   deployment's current version to `deprecated` and the previous approved version back to
   `production` in one transaction, emitting two `deployment.changed.v1` events; refused when
   there is no previous version. Both require an operator note.
7. **Signed artifacts only**: a `verify_artifact(manifest, path)` function checks the digest
   and MAC of every artifact file before any load, and `SpecialistProvider` refuses to load an
   artifact that fails it. Tampering test.
8. **CLI**: `make train SPEC=<file>`, `make promote MODEL=<version> TO=<state> NOTE=<text>`,
   `make rollback DEPLOYMENT=<id> NOTE=<text>`, `make models` listing registry ids, versions,
   states and evaluation ids only.
9. **Tests**: state machine exhaustively; approval required before training; policy revocation
   between approval and job start refuses the job; resumable job; reproducible artifacts;
   promotion requires a passed evaluation for that exact version and dataset version;
   rollback; tamper; no plaintext example content in any file under `models/` (assert on bytes);
   one `deployment.changed.v1` per tenant per transition; a CPU-only smoke test that runs the
   whole path train → evaluate → approve → shadow in under ten seconds, marked `smoke`.
10. **Docs**: `docs/runbooks/training-and-promotion.md`; update `pipelines/training/README.md`
    to describe what exists and what 3b adds; update `docs/status.md` milestone 3 row.

## Out of scope

Real LoRA training, PyTorch, PEFT, tokenizers, GPU tests (slice 3b, behind an optional
dependency group the reviewer will add). Shadow traffic execution, canary routing, router
training (milestone 4). Distillation (milestone 5). Do not modify `docs/spec/`,
`pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None.
