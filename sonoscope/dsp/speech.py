"""Linear-prediction, pitch, voice-activity, and LPC resynthesis utilities."""

from __future__ import annotations

import numpy as np


def lpc(signal: np.ndarray, order: int = 16) -> np.ndarray:
    """Estimate LPC coefficients with Levinson-Durbin recursion.

    Returns ``[1, a1, ..., a_order]`` for the prediction polynomial.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if order < 1 or x.size <= order:
        raise ValueError("order must be positive and smaller than signal length")
    x = x - np.mean(x)
    autocorr = np.correlate(x, x, mode="full")[x.size - 1:x.size + order]
    if autocorr[0] <= np.finfo(float).eps:
        return np.r_[1.0, np.zeros(order)]
    coeffs = np.zeros(order + 1, dtype=np.float64)
    coeffs[0] = 1.0
    error = autocorr[0]
    for i in range(1, order + 1):
        reflection = -(autocorr[i] + np.dot(coeffs[1:i], autocorr[i - 1:0:-1])) / error
        reflection = float(np.clip(reflection, -0.999, 0.999))
        previous = coeffs.copy()
        coeffs[i] = reflection
        if i > 1:
            coeffs[1:i] = previous[1:i] + reflection * previous[i - 1:0:-1]
        error *= 1.0 - reflection * reflection
        error = max(error, np.finfo(float).eps)
    return coeffs


def formants(
    signal: np.ndarray, sample_rate: int, order: int = 16,
    min_frequency: float = 90.0, max_frequency: float | None = None,
) -> np.ndarray:
    """Estimate vocal-tract formant frequencies from LPC polynomial roots."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    a = lpc(signal, order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    frequencies = np.angle(roots) * sample_rate / (2 * np.pi)
    bandwidths = -0.5 * (sample_rate / (2 * np.pi)) * np.log(np.maximum(np.abs(roots), 1e-12))
    upper = sample_rate / 2 if max_frequency is None else min(max_frequency, sample_rate / 2)
    valid = (frequencies >= min_frequency) & (frequencies <= upper) & (bandwidths < 500)
    return np.sort(frequencies[valid])


def estimate_pitch(
    signal: np.ndarray, sample_rate: int, min_hz: float = 60.0, max_hz: float = 400.0
) -> float:
    """Estimate fundamental frequency using normalized autocorrelation."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if sample_rate <= 0 or min_hz <= 0 or max_hz <= min_hz:
        raise ValueError("invalid sample rate or pitch range")
    if x.size < 3 or np.max(np.abs(x)) < 1e-10:
        return 0.0
    x = x - np.mean(x)
    x *= np.hanning(x.size)
    corr = np.correlate(x, x, mode="full")[x.size - 1:]
    lo = max(1, int(sample_rate / max_hz))
    hi = min(len(corr) - 1, int(sample_rate / min_hz))
    if hi <= lo:
        return 0.0
    segment = corr[lo:hi + 1]
    lag = lo + int(np.argmax(segment))
    if corr[lag] <= 0:
        return 0.0
    return float(sample_rate / lag)


def vad(
    signal: np.ndarray, frame_length: int = 400, hop_length: int = 160,
    threshold_db: float = -40.0,
) -> np.ndarray:
    """Return a boolean energy-based voice-activity decision per frame."""
    from .features import frame_signal

    frames = frame_signal(signal, frame_length, hop_length)
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    rms_db = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames * frames, axis=1)), 1e-12))
    return rms_db >= threshold_db


def lpc_resynthesis(excitation: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Filter an excitation signal through an LPC synthesis filter."""
    x = np.asarray(excitation, dtype=np.float64).reshape(-1)
    a = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if a.size == 0 or abs(a[0]) < np.finfo(float).eps:
        raise ValueError("coefficients must begin with a non-zero a0")
    a = a / a[0]
    try:
        from scipy.signal import lfilter

        return lfilter([1.0], a, x)
    except ImportError:
        y = np.zeros_like(x)
        for n in range(x.size):
            feedback = sum(a[k] * y[n - k] for k in range(1, min(len(a), n + 1)))
            y[n] = x[n] - feedback
        return y
