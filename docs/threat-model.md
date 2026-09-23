# Initial threat model

The scaffold has no inference or persistence path. The following are required design/test
work for later increments, not claims about implemented controls.

| Threat | Required control and evidence |
| --- | --- |
| Forged tenant or cross-tenant retrieval | Auth-derived tenant; document ACLs before retrieval; negative isolation tests |
| Prompt injection or poisoned retrieval | Source/instruction boundaries; restricted tools; indirect-injection evaluation |
| Secrets in logs or datasets | Pre-persistence and build-time scans; exception sanitisation; secret-leak tests |
| Training without consent | Current purpose policy and licences checked at build/job launch |
| Replay or duplicate jobs/events | Tenant-scoped idempotency, request fingerprint, immutable versions |
| Analytics outage/backpressure | Bounded queue, retry/dead letter, dropped-event counters; serving failure drills |
| Malicious artifacts | Scan, signature verification and immutable lineage before deployment |
| Model extraction/memorisation | Rate limits, private evaluation sets, canary extraction tests |
| Removal races | Tombstones and deletion watermarks consulted by late events and builders |
| Unapproved/unsafe model promotion | Role separation, hard quality/safety gates, audit and rollback |
| Adapter cross-tenant leakage (specialist trained on tenant A serves tenant B) | Dataset manifests carry the tenant set; registry refuses a deployment scope wider than training consent scope; per-tenant adapters or isolated serving where required; membership-inference canaries |
| Provider processes data outside permitted residency | `processing_region` on every route candidate; residency is a hard router constraint, tested with a negative routing test |
| Hash of short user input reversed by dictionary attack | User-derived hashes use keyed HMAC (`hash_scheme` recorded); plain SHA-256 only for governed knowledge content |

