"""Rank and context quality from actual pipeline retrieval, with ACL decoys."""

from collections import Counter
from statistics import fmean

from adaptive_llm.contracts import EvaluationSpecification, SuiteResult
from adaptive_llm.evaluation.runner import Case, Runner


class RetrievalSuite:
    async def run(
        self, runner: Runner, cases: list[Case], specification: EvaluationSpecification
    ) -> SuiteResult:
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        failures: Counter[str] = Counter()
        for case in cases:
            outcome = await runner.run(case)
            retrieval = outcome.retrieval
            candidates = retrieval.run.candidates[: case.k] if retrieval else []
            relevant = case.expected_citations
            found = [(c.document_id, c.chunk_id) in relevant for c in candidates]
            recalls.append(sum(found) / max(1, len(relevant)))
            reciprocal_ranks.append(next((1 / rank for rank, hit in enumerate(found, 1) if hit), 0))
            supplied = retrieval.supplied_chunks if retrieval else ()
            precisions.append(
                sum((c.document_id, c.chunk_id) in relevant for c in supplied)
                / max(1, len(supplied))
            )
            if any(
                c.tenant_id != case.tenant_id
                or "evaluation" not in c.allowed_applications
                or c.environment != "local"
                or c.index_id != case.request.rag.index_id
                for c in supplied
            ) or any(
                c.document_id
                in {
                    "synthetic-foreign",
                    "synthetic-private",
                    "synthetic-stale",
                    "synthetic-production",
                }
                for c in candidates
            ):
                failures["acl_isolation"] += 1
            if outcome.error or not relevant <= {(c.document_id, c.chunk_id) for c in candidates}:
                failures["retrieval_failure"] += 1
        return SuiteResult(
            suite="retrieval",
            items=len(cases),
            failures=dict(failures),
            metrics={
                "recall_at_k": fmean(recalls) if recalls else 0.0,
                "mrr": fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
                "context_precision": fmean(precisions) if precisions else 0.0,
                "acl_incidents": float(failures["acl_isolation"]),
            },
        )
