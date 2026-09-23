"""Paired percentile bootstrap; no unpaired or missing-data fallback."""

import math
import random
from collections.abc import Sequence
from statistics import NormalDist, fmean

from adaptive_llm.contracts import PairedComparison


def required_sample_size(margin: float, pilot_sd: float, confidence: float) -> int:
    if not (0 < margin < 1 and 0.5 < confidence < 1 and math.isfinite(pilot_sd) and pilot_sd >= 0):
        raise ValueError("invalid_sample_size_parameters")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    # Strictly below margin / 2, including the exact-integer boundary.
    return max(1, math.floor((2 * z * pilot_sd / margin) ** 2) + 1)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid_percentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    confidence: float = 0.95,
    seed: int = 23,
    resamples: int = 2000,
) -> PairedComparison:
    if len(candidate) != len(baseline):
        raise ValueError("unpaired_samples")
    if not 0.5 < confidence < 1 or resamples < 1:
        raise ValueError("invalid_bootstrap_parameters")
    deltas = [c - b for c, b in zip(candidate, baseline, strict=True)]
    if any(not math.isfinite(value) for value in deltas):
        raise ValueError("nonfinite_samples")
    if not deltas:
        return PairedComparison(sample_size=0)
    rng = random.Random(seed)
    samples = [fmean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples)]
    tail = (1 - confidence) / 2
    return PairedComparison(
        mean_delta=fmean(deltas),
        ci_lower=percentile(samples, tail),
        ci_upper=percentile(samples, 1 - tail),
        sample_size=len(deltas),
    )
