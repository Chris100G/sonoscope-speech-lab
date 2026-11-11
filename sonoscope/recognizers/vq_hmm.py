"""Discrete VQ-HMM recognizer for labeled feature sequences."""

from __future__ import annotations

import numpy as np

from sonoscope.dsp.hmm import DiscreteHMM, baum_welch
from sonoscope.dsp.quantization import lbg, quantize

from .dtw import TrainingData, _samples
from .types import RecognitionResult


class VQHMMRecognizer:
    """Quantize feature vectors and train one discrete HMM per class."""

    def __init__(
        self, codebook_size: int = 16, n_states: int = 5, iterations: int = 20,
        reject_threshold: float | None = None, seed: int = 0,
    ):
        if codebook_size <= 0 or n_states <= 0 or iterations <= 0:
            raise ValueError("codebook_size, n_states, and iterations must be positive")
        self.codebook_size = codebook_size
        self.n_states = n_states
        self.iterations = iterations
        self.reject_threshold = reject_threshold
        self.seed = seed
        self.codebook: np.ndarray | None = None
        self.models: dict[str, DiscreteHMM] = {}

    def fit(self, data: TrainingData) -> VQHMMRecognizer:
        """Fit a shared LBG codebook and class-conditional discrete HMMs."""
        examples = _samples(data)
        if not examples:
            raise ValueError("at least one training sequence is required")
        dimensions = {seq.shape[1] if seq.ndim == 2 else 1 for _, seq in examples}
        if len(dimensions) != 1:
            raise ValueError("all feature sequences must have the same dimension")
        vectors = np.concatenate([
            seq[:, None] if seq.ndim == 1 else seq for _, seq in examples
        ], axis=0)
        self.codebook = lbg(vectors, self.codebook_size)
        encoded: dict[str, list[np.ndarray]] = {}
        for label, sequence in examples:
            encoded.setdefault(label, []).append(quantize(sequence, self.codebook))
        self.models = {
            label: baum_welch(sequences, self.n_states, self.codebook_size,
                              self.iterations, seed=self.seed + idx)
            for idx, (label, sequences) in enumerate(sorted(encoded.items()))
        }
        return self

    def score(self, features: np.ndarray) -> dict[str, float]:
        """Return per-class log-likelihoods for a feature sequence."""
        if self.codebook is None:
            raise RuntimeError("fit must be called before scoring")
        sequence = np.asarray(features, dtype=float)
        if sequence.ndim == 1:
            sequence = sequence[:, None]
        symbols = quantize(sequence, self.codebook)
        return {label: model.score(symbols) for label, model in self.models.items()}

    def recognize(
        self, features: np.ndarray, reject_threshold: float | None = None
    ) -> RecognitionResult:
        """Classify by maximum log-likelihood; optionally reject below threshold."""
        scores = self.score(features)
        label = max(scores, key=scores.get)
        threshold = self.reject_threshold if reject_threshold is None else reject_threshold
        rejected = threshold is not None and scores[label] < threshold
        return RecognitionResult(None if rejected else label, scores[label], scores, rejected)

    def predict(self, features: np.ndarray) -> str | None:
        """Return the recognized label, or ``None`` when rejected."""
        return self.recognize(features).label
