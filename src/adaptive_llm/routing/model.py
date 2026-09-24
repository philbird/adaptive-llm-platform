"""Deterministic logistic suitability, independent Platt calibration and centroid OOD."""

import math
from collections.abc import Callable, Sequence
from statistics import fmean

from pydantic import Field

from adaptive_llm.contracts import Contract, RoutingFeatures, RoutingRow, SuiteResult
from adaptive_llm.gateway.identity import GatewayError


def categories(features: RoutingFeatures) -> list[str]:
    tokens = sum(features.input_tokens >= edge for edge in (32, 128, 512, 2048, 8192))
    score = sum(features.top_score >= edge for edge in (0.25, 0.5, 0.75))
    return [
        features.task,
        features.language,
        features.risk_tier,
        str(tokens),
        str(int(features.context_supplied)),
        str(score),
    ]


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


def fit(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[float],
    check: Callable[[], None],
    *,
    steps: int = 600,
) -> list[float]:
    if not vectors or len(vectors) != len(labels):
        raise GatewayError(422, "empty_training_split")
    weights = [0.0] * (len(vectors[0]) + 1)
    for step in range(steps):
        if step % 20 == 0:
            check()
        gradient = [0.0] * len(weights)
        for vector, label in zip(vectors, labels, strict=True):
            values = [1.0, *vector]
            error = sigmoid(sum(w * x for w, x in zip(weights, values, strict=True))) - label
            for index, value in enumerate(values):
                gradient[index] += error * value
        weights = [w - 0.2 * g / len(vectors) for w, g in zip(weights, gradient, strict=True)]
    return weights


def logit(weights: list[float], vector: list[float]) -> float:
    return sum(w * x for w, x in zip(weights, [1.0, *vector], strict=True))


def ece(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("calibration_samples_required")
    result = 0.0
    for index in range(10):
        pairs = [
            (p, y)
            for p, y in zip(probabilities, labels, strict=True)
            if min(9, int(p * 10)) == index
        ]
        if pairs:
            result += (
                len(pairs)
                / len(labels)
                * abs(fmean(p for p, _ in pairs) - fmean(y for _, y in pairs))
            )
    return result


def suitable(row: RoutingRow, version: str, threshold: float = 0.9) -> float:
    observation = row.candidates.get(version)
    foundation = row.candidates.get(row.foundation_id)
    return float(
        bool(
            observation
            and observation.validation_pass
            and observation.quality is not None
            and observation.quality >= threshold
            and foundation
            and foundation.quality is not None
            and observation.quality >= foundation.quality - 0.02
        )
    )


class Classifier(Contract):
    weights: list[float]
    calibration: list[float]
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    observed_cost_micros: int = Field(ge=0)


class Estimate(Contract):
    suitability: float
    confidence: float
    ood: float
    latency_ms: float


class LogisticRouter(Contract):
    architecture: str = "router-logistic-v1"
    vocabulary: list[list[str]]
    centroid: list[float]
    scales: list[float]
    classifiers: dict[str, Classifier]
    training_interactions: list[str]
    calibration_interactions: list[str]

    def vector(self, features: RoutingFeatures) -> list[float]:
        values: list[float] = []
        for category, known in zip(categories(features), self.vocabulary, strict=True):
            values.extend(float(category == value) for value in known)
            values.append(float(category not in known))
        return values

    def ood(self, features: RoutingFeatures) -> float:
        vector = self.vector(features)
        distance = math.sqrt(
            fmean(
                ((value - center) / scale) ** 2
                for value, center, scale in zip(vector, self.centroid, self.scales, strict=True)
            )
        )
        return distance / (1 + distance)

    def estimate(self, features: RoutingFeatures, version: str) -> Estimate:
        classifier = self.classifiers.get(version)
        if classifier is None:
            return Estimate(suitability=0, confidence=0, ood=1, latency_ms=0)
        value = logit(classifier.weights, self.vector(features))
        return Estimate(
            suitability=sigmoid(logit(classifier.calibration, [value])),
            confidence=classifier.confidence,
            ood=self.ood(features),
            latency_ms=classifier.latency_ms,
        )


def train_model(
    training: list[RoutingRow],
    calibration: list[RoutingRow],
    check: Callable[[], None],
) -> LogisticRouter:
    if not training or not calibration:
        raise GatewayError(422, "empty_training_split")
    if {r.interaction_id for r in training} & {r.interaction_id for r in calibration}:
        raise GatewayError(409, "routing_fold_overlap")
    vocabulary = [sorted({categories(r.features)[i] for r in training}) for i in range(6)]
    model = LogisticRouter(
        vocabulary=vocabulary,
        centroid=[],
        scales=[],
        classifiers={},
        training_interactions=sorted(r.interaction_id for r in training),
        calibration_interactions=sorted(r.interaction_id for r in calibration),
    )
    vectors = [model.vector(row.features) for row in training]
    centroid = [fmean(column) for column in zip(*vectors, strict=True)]
    scales = [
        max(0.25, math.sqrt(fmean((x - mean) ** 2 for x in column)))
        for mean, column in zip(centroid, zip(*vectors, strict=True), strict=True)
    ]
    classifiers: dict[str, Classifier] = {}
    versions = sorted({v for r in training for v in r.candidates if v != r.foundation_id})
    for version in versions:
        weights = fit(vectors, [suitable(row, version) for row in training], check)
        logits = [[logit(weights, model.vector(row.features))] for row in calibration]
        labels = [suitable(row, version) for row in calibration]
        platt = fit(logits, labels, check)
        probabilities = [sigmoid(logit(platt, x)) for x in logits]
        observations = [o for r in training if (o := r.candidates.get(version)) is not None]
        costs = [o.cost_micros for o in observations if o.cost_micros is not None]
        latencies = [o.latency_ms for o in observations if o.latency_ms is not None]
        classifiers[version] = Classifier(
            weights=weights,
            calibration=platt,
            confidence=(1 - ece(probabilities, labels)) * min(1, len(observations) / 4),
            latency_ms=fmean(latencies) if latencies else 0,
            observed_cost_micros=math.ceil(fmean(costs)) if costs else 0,
        )
    return model.model_copy(
        update={"centroid": centroid, "scales": scales, "classifiers": classifiers}
    )


def routing_suite(model: LogisticRouter, rows: list[RoutingRow]) -> SuiteResult:
    seen = set(model.training_interactions) | set(model.calibration_interactions)
    if any(row.interaction_id in seen for row in rows):
        raise GatewayError(409, "routing_fold_overlap")
    probabilities: list[float] = []
    labels: list[float] = []
    false_specialists = selected = unnecessary = possible = novel = detected = 0
    for row in rows:
        estimates = {v: model.estimate(row.features, v) for v in model.classifiers}
        eligible = [
            v
            for v, e in estimates.items()
            if e.suitability >= 0.9 and e.confidence >= 0.85 and e.ood <= 0.15
        ]
        chosen = (
            min(eligible, key=lambda v: (model.classifiers[v].observed_cost_micros, v))
            if eligible
            else None
        )
        selected += chosen is not None
        false_specialists += chosen is not None and not suitable(row, chosen)
        available = any(suitable(row, version) for version in estimates)
        possible += available
        unnecessary += available and chosen is None
        is_novel = any(
            c not in known
            for c, known in zip(categories(row.features), model.vocabulary, strict=True)
        )
        novel += is_novel
        detected += is_novel and model.ood(row.features) > 0.15
        for version, estimate in estimates.items():
            probabilities.append(estimate.suitability)
            labels.append(suitable(row, version))
    return SuiteResult(
        suite="routing",
        items=len(rows),
        completed=bool(rows and probabilities),
        metrics={
            "false_specialist_rate": false_specialists / selected if selected else 0,
            "unnecessary_foundation_rate": unnecessary / possible if possible else 0,
            "calibration_error": ece(probabilities, labels) if probabilities else 1,
            "ood_detection_rate": detected / novel if novel else 0,
            "ood_samples": float(novel),
            "specialist_selections": float(selected),
        },
    )
