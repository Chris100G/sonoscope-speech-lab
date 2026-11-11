"""Feature-vector distances and dynamic time warping."""

from __future__ import annotations

import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between equal-shaped vectors."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.linalg.norm(x - y))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance, defining distance between zero vectors as 0."""
    x, y = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("vectors must have the same shape")
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(1 - np.dot(x, y) / denominator)


def cepstral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean cepstral distance for equal-length coefficient vectors."""
    x, y = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("cepstral vectors must have the same shape")
    return float(np.sqrt(2.0 * np.sum((x - y) ** 2)))


def itakura_distance(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return Itakura-Saito divergence between positive spectral envelopes."""
    p = np.asarray(reference, dtype=float).reshape(-1)
    q = np.asarray(estimate, dtype=float).reshape(-1)
    if p.shape != q.shape:
        raise ValueError("spectral envelopes must have the same shape")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("spectral envelopes must be non-negative")
    floor = np.finfo(float).tiny
    ratio = np.maximum(p, floor) / np.maximum(q, floor)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def dtw(
    sequence_a: np.ndarray, sequence_b: np.ndarray, metric: str = "euclidean"
) -> tuple[float, list[tuple[int, int]]]:
    """Compute classic DTW and return normalized cost and an alignment path."""
    a, b = np.asarray(sequence_a, dtype=float), np.asarray(sequence_b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("sequences must be 2-D with matching feature dimensions")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("sequences must not be empty")
    distance_fn = {
        "euclidean": euclidean_distance,
        "cosine": cosine_distance,
        "cepstral": cepstral_distance,
        "itakura": itakura_distance,
    }.get(metric)
    if distance_fn is None:
        raise ValueError("metric must be 'euclidean', 'cosine', 'cepstral', or 'itakura'")
    costs = np.full((len(a) + 1, len(b) + 1), np.inf)
    costs[0, 0] = 0
    steps = np.zeros((len(a), len(b)), dtype=np.uint8)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            predecessors = (costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1])
            choice = int(np.argmin(predecessors))
            costs[i, j] = distance_fn(a[i - 1], b[j - 1]) + predecessors[choice]
            steps[i - 1, j - 1] = choice
    i, j = len(a) - 1, len(b) - 1
    path = [(i, j)]
    while i or j:
        direction = steps[i, j]
        if i > 0 and j > 0 and direction == 0:
            i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or direction == 1):
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    path.reverse()
    return float(costs[-1, -1] / len(path)), path


def dtw_distance(
    sequence_a: np.ndarray, sequence_b: np.ndarray, metric: str = "euclidean"
) -> float:
    """Return only the normalized DTW cost."""
    return dtw(sequence_a, sequence_b, metric)[0]
