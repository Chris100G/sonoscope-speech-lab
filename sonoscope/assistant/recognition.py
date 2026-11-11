"""Recognizer score fusion and confidence-based rejection."""

from __future__ import annotations

import numpy as np

from sonoscope.recognizers.types import RecognitionResult


def fuse_scores(
    score_sets: dict[str, dict[str, float]], weights: dict[str, float] | None = None
) -> dict[str, float]:
    """Fuse score maps after per-recognizer z-normalization.

    Missing labels contribute zero; all returned scores are higher-is-better.
    """
    if not score_sets:
        return {}
    labels = sorted({label for scores in score_sets.values() for label in scores})
    combined = {label: 0.0 for label in labels}
    total_weight = 0.0
    for name, scores in score_sets.items():
        weight = 1.0 if weights is None else float(weights.get(name, 0.0))
        if weight < 0:
            raise ValueError("fusion weights cannot be negative")
        if weight == 0 or not scores:
            continue
        values = np.asarray(list(scores.values()), dtype=float)
        mean, std = float(values.mean()), float(values.std())
        normalized = {label: ((float(value) - mean) / std if std > 1e-12 else 0.0)
                      for label, value in scores.items()}
        for label in labels:
            combined[label] += weight * normalized.get(label, 0.0)
        total_weight += weight
    if total_weight:
        combined = {label: value / total_weight for label, value in combined.items()}
    return combined


def reject(
    scores: dict[str, float], threshold: float = 0.0
) -> RecognitionResult:
    """Select the highest score unless it is below ``threshold``."""
    if not scores:
        return RecognitionResult(None, float("-inf"), {}, True)
    label = max(scores, key=scores.get)
    score = float(scores[label])
    rejected = score < threshold
    return RecognitionResult(None if rejected else label, score, scores, rejected)


def recognize_with_fusion(
    score_sets: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    threshold: float = 0.0,
) -> RecognitionResult:
    """Fuse multiple recognizer outputs and apply open-set rejection."""
    return reject(fuse_scores(score_sets, weights), threshold)
