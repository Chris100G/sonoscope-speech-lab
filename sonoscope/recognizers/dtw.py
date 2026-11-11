"""Template speech recognizer based on dynamic time warping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from sonoscope.dsp.distances import dtw_distance

from .types import RecognitionResult

TrainingData = Mapping[str, object] | Sequence[tuple[str, np.ndarray]]


def _samples(data: TrainingData) -> list[tuple[str, np.ndarray]]:
    if isinstance(data, Mapping):
        result = []
        for label, value in data.items():
            try:
                packed = np.asarray(value, dtype=float)
            except ValueError:
                packed = np.asarray(value, dtype=object)
            if packed.ndim == 2 and packed.dtype != object:
                sequences = [packed]
            elif packed.ndim == 3 and packed.dtype != object:
                sequences = list(packed)
            elif isinstance(value, (list, tuple)) and value:
                sequences = [np.asarray(sequence, dtype=float) for sequence in value]
            else:
                sequences = [np.asarray(value, dtype=float)]
            result.extend((str(label), sequence) for sequence in sequences)
        return result
    result = []
    for item in data:
        if len(item) != 2:
            raise ValueError("training items must be (label, feature_sequence) pairs")
        result.append((str(item[0]), np.asarray(item[1], dtype=float)))
    return result


class DTWTemplateRecognizer:
    """Nearest-template classifier for variable-length feature sequences."""

    def __init__(self, metric: str = "euclidean", reject_threshold: float | None = None):
        self.metric = metric
        self.reject_threshold = reject_threshold
        self.templates: dict[str, list[np.ndarray]] = {}

    def fit(self, data: TrainingData) -> DTWTemplateRecognizer:
        """Store labeled templates; ``data`` is a mapping or iterable of pairs."""
        templates: dict[str, list[np.ndarray]] = {}
        for label, sequence in _samples(data):
            if sequence.ndim == 1:
                sequence = sequence[:, None]
            if sequence.ndim != 2 or sequence.shape[0] == 0:
                raise ValueError("each template must be a non-empty feature sequence")
            templates.setdefault(label, []).append(sequence)
        if not templates:
            raise ValueError("at least one template is required")
        self.templates = templates
        return self

    def score(self, features: np.ndarray) -> dict[str, float]:
        """Return negative best DTW distance per label (higher is better)."""
        query = np.asarray(features, dtype=float)
        if query.ndim == 1:
            query = query[:, None]
        if query.ndim != 2 or query.shape[0] == 0:
            raise ValueError("features must be a non-empty sequence")
        return {
            label: -min(dtw_distance(query, template, self.metric) for template in samples)
            for label, samples in self.templates.items()
        }

    def recognize(
        self, features: np.ndarray, reject_threshold: float | None = None
    ) -> RecognitionResult:
        """Return the best label, rejecting it when its distance is too large."""
        scores = self.score(features)
        label = max(scores, key=scores.get)
        threshold = self.reject_threshold if reject_threshold is None else reject_threshold
        rejected = threshold is not None and -scores[label] > threshold
        return RecognitionResult(None if rejected else label, scores[label], scores, rejected)

    def predict(self, features: np.ndarray) -> str | None:
        """Return the recognized label, or ``None`` for a rejected utterance."""
        return self.recognize(features).label
