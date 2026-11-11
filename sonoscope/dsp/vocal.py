"""Compatibility exports for vocal-analysis and LPC helpers."""

import numpy as np

from .speech import estimate_pitch, formants, lpc, lpc_resynthesis, vad


def formant(
    signal: np.ndarray,
    sample_rate: int,
    order: int = 16,
    min_frequency: float = 90.0,
    max_frequency: float | None = None,
) -> np.ndarray:
    """Return estimated formant frequencies (alias for :func:`formants`)."""
    return formants(signal, sample_rate, order, min_frequency, max_frequency)


__all__ = ["estimate_pitch", "formant", "formants", "lpc", "lpc_resynthesis", "vad"]
