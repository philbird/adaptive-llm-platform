# Decisions required before production implementation

All owners below are roles to assign, not assumed approvals. Local examples are synthetic only.

| Decision | Owner to assign | Current safe position |
| --- | --- | --- |
| First task, application, language and risk | Product / ML | Synthetic English refund-policy examples only |
| Tenancy and identity; application ACLs | Platform / security | No user-content endpoints exposed |
| Lawful basis, separate purposes and review permission | Data / privacy | Retained logging, evaluation and training disabled |
| Regions, retention and deletion SLA | Data / platform | No content persistence; production values unset |
| Foundation/base model, licence, hosting, contracts | ML / procurement | No provider calls or model download |
| RAG sources, document ACLs and freshness | Knowledge / security | Synthetic fixtures only |
| Throughput, token sizes and latency/availability SLOs | Platform / product | No production claims |
| Rubrics, quality margins and critical segments | Product / ML | Spec's example thresholds recorded as provisional |
| Human review and promotion authority | Data / ML / security / product | No promotion API |
| Events, warehouse, workflow, registry and secrets | Platform | Interfaces planned; vendors not selected |
| Canary, recovery targets and rollback authority | Platform / product | No deployed specialists |

