"""Bounded concurrent serving requests, including persistence and integer-micro cost."""

import asyncio
from time import perf_counter

from adaptive_llm.contracts import EvaluationSpecification, SuiteResult
from adaptive_llm.evaluation.runner import Case, Outcome, Runner
from adaptive_llm.evaluation.stats import percentile
from adaptive_llm.evaluation.suites.scoring import assertions


class PerformanceSuite:
    async def run(
        self, runner: Runner, cases: list[Case], specification: EvaluationSpecification
    ) -> SuiteResult:
        if not cases:
            return SuiteResult(suite="performance", items=0, metrics={}, completed=False)
        semaphore = asyncio.Semaphore(specification.concurrency)

        async def one(index: int) -> tuple[Outcome, bool]:
            async with semaphore:
                case = cases[index % len(cases)]
                outcome = await runner.run(case)
                return outcome, all(assertions(case, outcome).values())

        start = perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(specification.performance_requests)))
        elapsed = perf_counter() - start
        latencies = [outcome.latency_ms for outcome, _ in results]
        successes = sum(success for _, success in results)
        total = sum(outcome.cost_micros for outcome, _ in results)
        return SuiteResult(
            suite="performance",
            items=len(results),
            total_cost_micros=total,
            cost_per_success_micros=(total + successes - 1) // successes if successes else None,
            metrics={
                "p50_latency_ms": percentile(latencies, 0.50),
                "p95_latency_ms": percentile(latencies, 0.95),
                "p99_latency_ms": percentile(latencies, 0.99),
                "error_rate": sum(outcome.error for outcome, _ in results) / len(results),
                "success_rate": successes / len(results),
                "requests_per_second": len(results) / max(elapsed, 1e-9),
                "concurrency": float(specification.concurrency),
            },
        )
