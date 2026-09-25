# Decisions required before production implementation

All owners below are roles to assign, not assumed approvals. Local examples are synthetic only.

Three decisions gate milestone 1. To avoid stalling on owners, each has a **provisional,
reversible** default recorded here; the owner may overturn it before any production use.

| Decision | Owner to assign | Current safe position | Provisional default for milestone 1 |
| --- | --- | --- | --- |
| First task, application, language and risk | Product / ML | Synthetic English refund-policy examples only | Task `customer_support.refund_policy`, application `synthetic-app`, English, risk tier medium |
| Tenancy and identity; application ACLs | Platform / security | No user-content endpoints exposed | Static API-key to tenant/application map from local config; OIDC later behind the same interface |
| Foundation/base model, licence, hosting, contracts | ML / procurement | No provider calls or model download | Deterministic fake provider only; first real provider chosen by ADR |
| Lawful basis, separate purposes and review permission | Data / privacy | Retained logging, evaluation and training disabled | Not required for milestone 1 |
| Regions, retention and deletion SLA | Data / platform | No content persistence; production values unset | Not required for milestone 1 |
| RAG sources, document ACLs and freshness | Knowledge / security | Synthetic fixtures only | Not required for milestone 1 |
| Throughput, token sizes and latency/availability SLOs | Platform / product | No production claims | Not required for milestone 1 |
| Rubrics, quality margins and critical segments | Product / ML | Spec's example thresholds recorded as provisional | Not required for milestone 1 |
| Human review and promotion authority | Data / ML / security / product | No promotion API | Not required for milestone 1 |
| Events, warehouse, workflow, registry and secrets | Platform | Interfaces planned; vendors not selected | Not required for milestone 1 |
| Canary, recovery targets and rollback authority | Platform / product | No deployed specialists | Not required for milestone 1 |


## Recorded answers

Answers from the owner replace the provisional defaults above from the date shown.

| Date | Decision | Answer | How it is applied |
| --- | --- | --- | --- |
| 2026-09-25 | 2. Lawful basis, consent and training permission | Phil: proceed without a separate lawful-basis exercise for the first tenant. | The first tenant is Phil's own ManyFails editorial pipeline. Its inputs are public web pages and operator-authored candidates; there are no end users and no personal data beyond names already published in the press. All purposes (processing, retained logging, evaluation, human review, training) are enabled for that tenant in its policy file. Anything a member of the public types into ManyFails (the idea check) is excluded from the platform because ManyFails promises that nothing typed is stored. Revisit before any tenant with end users. |
