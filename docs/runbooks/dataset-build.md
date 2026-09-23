# Local feedback and dataset builds (slice 2a)

The default serving policy still denies training and retained content logging. Dataset
examples require logging refs; operational replay is never a training source. Use the
separate, explicitly selected synthetic policy below. It permits logging/training for
`synthetic-a` and logging without training for `synthetic-b`. Existing rows do not acquire
missing refs when policy changes.

Start a local demo server from the repository root:

```sh
uv run --locked python -c 'from pathlib import Path; import uvicorn; from adaptive_llm.app import Settings, create_app; uvicorn.run(create_app(Settings(data_dir=Path(".local/dataset-demo"), policy_path=Path("configs/policy/dataset-demo.json"))), host="127.0.0.1", port=8000, access_log=False)'
```

In another terminal, create one synthetic interaction and an explicitly authorised correction:

```sh
INTERACTION=$(curl -fsS http://127.0.0.1:8000/v1/inference \
  -H 'Authorization: Bearer synthetic-key-a' -H 'X-Subject: synthetic-dataset-caller' \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"synthetic-dataset-demo","application_id":"support-assistant","messages":[{"role":"user","content":"SYNTHETIC: unused items receipt return window"}],"rag":{"enabled":true,"index_id":"synthetic-kb"}}' \
  | uv run --locked python -c 'import json,sys; print(json.load(sys.stdin)["interaction_id"])')

curl -fsS "http://127.0.0.1:8000/v1/interactions/$INTERACTION/feedback" \
  -H 'Authorization: Bearer synthetic-key-a' -H 'X-Subject: synthetic-reviewer' \
  -H 'Content-Type: application/json' \
  -d '{"label_type":"thumb","value":{"score":0,"max_score":1}}'

curl -fsS "http://127.0.0.1:8000/v1/interactions/$INTERACTION/correction" \
  -H 'Authorization: Bearer synthetic-key-a' -H 'X-Subject: synthetic-reviewer' \
  -H 'Content-Type: application/json' \
  -d '{"correction":"SYNTHETIC: Example Shop accepts unused items within 30 days with a receipt.","training_authorised":true}'

curl -fsS http://127.0.0.1:8000/v1/datasets/builds \
  -H 'Authorization: Bearer synthetic-operator-key' -H 'Content-Type: application/json' \
  --data-binary @configs/datasets/synthetic.json
```

Feedback responses contain `source="user"`, a pseudonymous actor, ids and encrypted refs.
They never echo comments or corrections. The correction endpoint requires an explicit JSON
boolean `training_authorised`. Generic feedback labelled `correction` is a label only; the
dedicated correction endpoint creates a correction target. Ref storage uses the current
logging policy and persistence redactor. Redaction failure records a fixed error with null
refs; it cannot put content in storage. Cross-tenant requests return 404; deleted/expired
interactions return 409. Labels, interaction links, payloads and `feedback.recorded.v1`
commit together. Privacy deletion and retention remove these refs and payloads too.

The build returns a content-free manifest with a UUIDv7 `version`, counts
`{"train":1,"validation":0,"test":0}`, target label mix `{"correction":1}`, a digest,
and `approval.status="pending"`. This is a smoke dataset; it is not approved for training.
The declared source window is start-inclusive/end-exclusive. Adjust the window for later
dates. The policy retains source content for one hour, so build before its expiry.

Retrieve the manifest using its returned dataset id and version:

```sh
curl -fsS http://127.0.0.1:8000/v1/datasets/synthetic-support/versions/VERSION \
  -H 'Authorization: Bearer synthetic-operator-key'
```

Both dataset APIs require an operator key. The key's configured `dataset_tenants` grant
access; request fields cannot grant access. User keys receive 403. Operators with insufficient
tenant grants cannot build or read the manifest. The synthetic operator grants both local
tenants. Approval remains pending; no approval endpoint exists.

Stop the demo server before the CLI build, then run the same builder against its stored data:

```sh
make dataset-build SPEC=configs/datasets/synthetic.json DATA_DIR=.local/dataset-demo POLICY=configs/policy/dataset-demo.json
```

The CLI prints the manifest as JSON and leaves its build events pending in the outbox. It uses
the configured local synthetic operator key, or accepts `--operator-key` via
`uv run --locked python -m adaptive_llm.datasets`. `PAYLOAD_KEYRING` works as for serving.
Without `POLICY`, the CLI uses the default policy, which denies training; the demo spec then
fails the policy-version check. Failure exits 1 with `dataset_build_failed` and no content.

Each successful build creates a new immutable directory:

```text
.local/dataset-demo/datasets/synthetic-support/<version>/
  synthetic-a.train.jsonl.enc
  synthetic-a.validation.jsonl.enc
  synthetic-a.test.jsonl.enc
  synthetic-b.train.jsonl.enc
  synthetic-b.validation.jsonl.enc
  synthetic-b.test.jsonl.enc
  manifest.json
  manifest.mac
  data-card.md
```

Each shard is a JSON envelope containing base64 AES-GCM ciphertext, nonce, key version,
tenant, split, field `dataset` and its plaintext SHA-256. Its decrypted bytes are JSONL.
AAD is exactly `tenant|dataset_id/version|split`. No plaintext JSONL temp files are created.
The manifest's content digest is SHA-256 of the concatenated lowercase hex plaintext hashes,
ordered by sorted tenant id, then `train`, `validation`, `test`, including empty shards.
The detached MAC is HMAC-SHA256 over the exact UTF-8 manifest bytes, using the
`dataset-manifest-v1` purpose-derived key. Production replaces it with an asymmetric signature
from a signing key the builder does not hold. The data card contains counts, exclusion reasons,
target mix and limitations only, including this production signing requirement.

Migration 0004 stores the manifest; migration 0005 backfills `interactions.started_at` from
the stored JSON and indexes `(tenant_id, started_at)` for SQL source-window selection.
Publishing enqueues one `dataset.built.v1` per specification tenant with the same trace id
and distinct event ids. Each tenant sees its dataset lifecycle, including tenants with zero
eligible examples. Each event contains the same manifest digest. A failed write or rename
rolls back SQL and removes that build's artifacts.
Manifests become API-visible only after commit. An abrupt process crash can leave an
unregistered encrypted directory; it has no committed manifest or successful build event
and must not be consumed. UUID versions are never reused. The build runs on a worker thread
in three phases:

1. A short read transaction takes the build-start watermark and copies eligible metadata,
   feedback and needed encrypted payload blobs into build-local memory.
2. Decryption, construction, indexed deduplication, splitting, encryption and staging-file
   writes run without the database lock. Inference can persist while this phase is running.
3. A short publish transaction checks the selected interactions and subjects for new tombstones,
   inserts the manifest and tenant events, then renames the staging directory to the final version.

If publication detects a deletion, the affected candidates are excluded and counted once each
as `deleted_during_build`. Curation, splitting and staging repeat outside the lock, followed by
another publish check. This allows a surviving duplicate to replace its deleted exemplar and
re-checks the minimum train count. If the minimum cannot be met, nothing publishes. The original
watermark stays fixed. Policy and feedback otherwise reflect the read snapshot. This remains a
local factory rather than a distributed job runner.

## Selection and reproducibility

Selection re-evaluates `PolicyEngine` for the trusted stored tenant/application/subject.
It requires current training permission (plus evaluation permission for evaluation purpose),
completed status, active retention, no interaction/subject tombstone, successful persistence
redaction, available logging refs and `synthetic`/`internal-approved` supplied licences.
Declared and current policy versions must match. Stored training booleans are ignored.
Each exclusion reason is counted; one interaction may contribute multiple reasons.

An unresolved negative is a user thumb/rubric score strictly below half scale without a
later authorised, persisted correction. This deliberately treats unlicensed or unavailable
corrections as unresolved. A later negative reopens the exclusion. Resolution is positive
strictly above half scale. The default target preference is authorised correction, then
validated production output with positive resolution, then validated production output.
Production targets must finish normally, pass validation and have no tool calls. Missing,
corrupt, unauthentic or hash-mismatched content fails closed. A configured preferred correction
that cannot be read is excluded rather than silently replaced with a different target.

Construction re-runs persistence redaction, including on source text. Exact source document,
chunk and index versions and content hash must resolve under tenant, environment, current
residency and application ACL. Source text is enclosed in
`<<source document_id/chunk_id version>>…<</source>>`; delimiter-like source text is escaped.
Examples contain placeholder system/policy instructions, conversation, ordered sources,
an empty tool-result list, target/source, policy/code versions, provenance and basic labels.
No raw examples appear in manifests, cards, events or CLI output.

Exact dedup uses a separate keyed hash of canonical input plus target. Near dedup uses
NFKC/casefold word 5-gram Jaccard strictly greater than `near_duplicate_threshold`, keeping
the earliest by interaction time/id. Very short text uses its complete token tuple. The
same comparison removes overlap with the 20 synthetic golden items in
`tests/fixtures/golden/`. An inverted shingle index limits accepted-example comparisons to
examples sharing at least one shingle, preserving the exhaustive algorithm's results.
Highly overlapping corpora can still require quadratic comparisons.

Connected components join both document-family and subject keys, including transitive
overlap. Missing subjects do not all form one group. Seeded component shuffling aims for
80/10/10 splits; large components can skew ratios. Families are shared grouping identifiers
across the tenant set; subject HMACs are already tenant scoped. Optional `time_split` takes
UTC `train_end` and `validation_end`: each whole component follows its latest example.
The build rejects a train split below `minimum_examples`.

Rebuilding the same specification and eligible snapshot produces identical ordered JSONL,
content digest and stable manifest fields. Ciphertext differs because AES-GCM uses fresh
nonces. The task's identical-manifest requirement conflicts with its mandatory new versions
and build-start watermarks: tests exclude exactly `version`, `created_at` and
`deletions_applied_through` from manifest equality. The detached MAC also changes.
`transformation_code_revision` is `git rev-parse HEAD`, or `unknown` if unavailable. It is
captured once at application startup and reused by every build on that builder.

`deletions_applied_through` records build start. Delete a subject with the existing privacy
API, then rebuild: every interaction for that pseudonym is absent and the watermark advances.
Existing immutable artifacts are not rewritten; external deletion propagation, dataset
revocation, key rotation of filesystem artifacts and artifact backup/reconciliation are
outside this slice. Existing database backup/rotation tools continue to cover SQLite only.

## Checks

```sh
make contracts
make check integration
uv run --locked pytest tests/security tests/load -s
make drills
```

The `dataset_seed` fixture in `tests/conftest.py` runs 60 synthetic interactions across two
tenants with mixed feedback/corrections, source families, shared subjects, a subject deletion
and denied training for tenant B. Tests verify selection, all target preferences, atomic
writes, encryption bindings, source isolation, benchmark filtering, transitive grouping,
time splits, CLI/API parity, reproducibility, deletion rebuilds, concurrent inference during a
blocked constructor, deletion races at publication, SQL index/backfill and indexed-dedup
equivalence. No dependency was added.
