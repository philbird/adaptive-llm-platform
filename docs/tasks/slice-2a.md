# Task: slice 2a — feedback, labels and the dataset factory

Milestone 2 begins. Implement the first half as described in the specification (`docs/spec/`,
sections 7.9, 8.5, 8.6, 9.2, 9.3 dataset build only, 10.4, 11.1, 11.2, 20.3, 21.2 step 8,
21.4) and `pipelines/dataset_builder/README.md`. Read `AGENTS.md`, `docs/status.md` and
`docs/runbooks/interaction-walkthrough.md` first. Milestone 1 is on `main`; do not change its
serving behaviour except where this brief says so. Everything remains local and synthetic.

## Deliverables

1. **Feedback endpoint.** `POST /v1/interactions/{interaction_id}/feedback` taking
   `FeedbackInput`, authenticated, tenant-scoped (404 for another tenant's interaction),
   refused for tombstoned interactions. Persist a `Feedback` record with
   `source="user"`, the actor pseudonymised like the subject, `comment` stored only as an
   encrypted payload ref when content logging is allowed, and `training_authorised` copied
   from the input. Append the feedback id to the interaction's `feedback_ids` and emit
   `feedback.recorded.v1` through the outbox in the same transaction. Also
   `POST /v1/interactions/{interaction_id}/correction`: a `correction` body field (bounded
   `Content`) stored as a payload ref with field `correction`, `label_type="correction"`,
   and `training_authorised` required to be explicit. Tests for isolation, tombstones, refs
   null when logging is disallowed, and the event.
2. **Dataset contracts** in `contracts.py` (then `make contracts`): `DatasetSpecification`
   (dataset id, purpose, tenant set, source window, eligibility policy version, target
   preference order, split strategy, grouping keys, minimum examples, seed) and
   `DatasetManifest` matching spec 8.6 (version, counts per split, split strategy, content
   digest, `deletions_applied_through`, quality summary, approval status, transformation code
   revision from `git rev-parse HEAD` or `"unknown"` when unavailable). Both are `Record`s.
3. **Eligibility** in `src/adaptive_llm/datasets/eligibility.py`: for a tenant and window,
   select interactions where the *current* tenant policy allows training (re-evaluate via the
   `PolicyEngine`, never the stored boolean), `status == "completed"`, no interaction or
   subject tombstone, not expired, `licence_class` of every supplied chunk is
   `synthetic` or `internal-approved`, no unresolved negative feedback (any user thumb or
   rubric below half scale without a later correction), and the persistence redactor did not
   fail. Every exclusion is counted by reason code. Output is a list of `Example` candidates
   holding only refs and hashes at this stage.
4. **Example construction** per spec 11.1: system and policy instructions placeholder, the
   redacted conversation decrypted from its payload ref, supplied RAG chunks with explicit
   source boundaries (`<<source document_id/chunk_id version>>…<</source>>`), permitted tool
   results (none yet), and the target chosen by the preference order: correction with
   `training_authorised` → response with positive resolution feedback → production output that
   passed validation. Record `target_source` on each example. Examples are built only in
   memory and written to the dataset shard, never logged.
5. **Deduplication and decontamination**: exact dedup on the keyed hash of input+target; near
   dedup by 5-gram Jaccard over normalised text above a configurable threshold, keeping the
   earliest; benchmark decontamination against `tests/fixtures/golden/` (create a small
   synthetic golden set of 20 items) by the same near-dedup rule. Count removals by reason.
6. **Grouped splits**: split by `document_family` and by subject pseudonym jointly (connected
   components over the two keys) so no family or subject appears in more than one split; then by
   time if a `time_split` is configured. Deterministic from the specification seed. Reject a
   build whose train split is below `minimum_examples`. A test constructs overlapping
   families and asserts zero leakage across splits.
7. **Shards and manifest**: write JSONL shards per split under
   `<data_dir>/datasets/<dataset_id>/<version>/`, encrypted with the payload cipher (field
   `dataset`, AAD `tenant|dataset_id/version|split`), a `manifest.json` (plaintext, content
   free) and a data card markdown with counts, exclusion reasons, label mix and known
   limitations. Content digest is SHA-256 over the ordered shard ciphertexts' plaintext
   hashes. Versions are immutable: rebuilding with the same id creates a new version and never
   overwrites. `deletions_applied_through` is the clock at build start; a deletion request
   after that time must be reflected in the next build (test: delete a subject, rebuild,
   assert the example is gone and the manifest shows the new watermark).
8. **Build API and CLI**: `POST /v1/datasets/builds` (control plane: requires a second bearer
   key class `operator` in `configs/identity/local.json`; user keys get 403) taking a
   `DatasetSpecification`, running the build off the event loop, persisting the manifest in a
   new `dataset_manifests` table (migration 0004) and emitting `dataset.built.v1`.
   `GET /v1/datasets/{dataset_id}/versions/{version}` returns the manifest. `make
   dataset-build SPEC=<file>` runs the same build from the CLI. Approval stays `pending`; there
   is no approval API yet.
9. **Reproducibility test**: build twice from the same specification and stored data; assert
   identical manifests except `created_at` and identical content digests.
10. **Seed data for tests**: a fixture helper that runs 60 synthetic interactions across the two
    tenants with mixed feedback, corrections, a subject deletion and one tenant whose policy
    denies training, so eligibility, targets and splits are all exercised.
11. **Docs**: `docs/runbooks/dataset-build.md` with commands and expected output;
    update `pipelines/dataset_builder/README.md` to describe what exists; update
    `docs/status.md` milestone 2 row.

## Out of scope

Evaluation harness, baseline locking, human review UI, training, approval API, registry
(slice 2b and milestone 3). Real providers, streaming. Do not modify `docs/spec/`,
`pyproject.toml` or `uv.lock`.

## Allowed dependency additions

None. Use the standard library for hashing, n-grams and JSONL.
