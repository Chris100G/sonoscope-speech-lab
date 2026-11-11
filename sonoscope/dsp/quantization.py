"""Linde-Buzo-Gray vector quantization and nearest-centroid encoding."""

from __future__ import annotations

import numpy as np


def _as_vectors(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("vectors must be a non-empty (samples, dimensions) array")
    return x


def quantize(vectors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Map each vector to its nearest codebook entry and return integer indices."""
    x, centers = _as_vectors(vectors), _as_vectors(codebook)
    if x.shape[1] != centers.shape[1]:
        raise ValueError("vectors and codebook must have equal dimensions")
    distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def lbg(
    vectors: np.ndarray, codebook_size: int, epsilon: float = 0.01,
    max_iterations: int = 100,
) -> np.ndarray:
    """Train a vector-quantizer codebook using split-and-refine LBG."""
    x = _as_vectors(vectors)
    if codebook_size <= 0 or epsilon <= 0 or max_iterations <= 0:
        raise ValueError("codebook_size, epsilon, and max_iterations must be positive")
    centers = x.mean(axis=0, keepdims=True)
    while len(centers) < codebook_size:
        perturbation = epsilon * np.maximum(np.abs(centers), 1.0)
        centers = np.vstack((centers + perturbation, centers - perturbation))[:codebook_size]
        for _ in range(max_iterations):
            labels = quantize(x, centers)
            updated = centers.copy()
            for idx in range(len(centers)):
                members = x[labels == idx]
                if members.size:
                    updated[idx] = members.mean(axis=0)
            if np.allclose(updated, centers, atol=1e-8, rtol=1e-6):
                centers = updated
                break
            centers = updated
    return centers
