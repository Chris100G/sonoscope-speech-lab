"""Short-time spectral features: framing, mel filters, MFCCs, and deltas."""

from __future__ import annotations

import numpy as np


def pre_emphasis(signal: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """Apply first-order pre-emphasis while preserving the first sample."""
    x = np.asarray(signal, dtype=np.float64)
    if not 0 <= coefficient < 1:
        raise ValueError("coefficient must be in [0, 1)")
    if x.size == 0:
        return x.copy()
    return np.concatenate((x[:1], x[1:] - coefficient * x[:-1]))


def frame_signal(
    signal: np.ndarray, frame_length: int, hop_length: int, pad: bool = True
) -> np.ndarray:
    """Split a signal into overlapping frames; optionally pad its final frame."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if x.size == 0:
        return np.empty((0, frame_length), dtype=np.float64)
    if pad:
        count = max(1, int(np.ceil(max(0, x.size - frame_length) / hop_length)) + 1)
        needed = (count - 1) * hop_length + frame_length
        x = np.pad(x, (0, max(0, needed - x.size)))
    elif x.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    count = 1 + (x.size - frame_length) // hop_length
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None]
    return x[indices]


def window_frames(frames: np.ndarray, window: str = "hamming") -> np.ndarray:
    """Apply a named analysis window to each frame."""
    data = np.asarray(frames, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("frames must have shape (frames, samples)")
    choices = {
        "hamming": np.hamming,
        "hann": np.hanning,
        "blackman": np.blackman,
    }
    if window not in choices:
        raise ValueError("window must be 'hamming', 'hann', or 'blackman'")
    return data * choices[window](data.shape[1])[None, :]


def _mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 26,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Create triangular filters over the one-sided FFT frequency bins."""
    if sample_rate <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("sample_rate, n_fft, and n_mels must be positive")
    upper = sample_rate / 2 if f_max is None else float(f_max)
    if f_min < 0 or upper <= f_min or upper > sample_rate / 2:
        raise ValueError("require 0 <= f_min < f_max <= Nyquist")
    mel_points = np.linspace(_mel(f_min), _mel(upper), n_mels + 2)
    bins = np.floor((n_fft + 1) * _hz(mel_points) / sample_rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1:m + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, bank.shape[1])):
            bank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, bank.shape[1])):
            bank[m - 1, k] = (right - k) / (right - center)
    return bank


def _dct_type_2(values: np.ndarray, count: int) -> np.ndarray:
    try:
        from scipy.fft import dct

        return dct(values, type=2, axis=-1, norm="ortho")[..., :count]
    except ImportError:
        n = values.shape[-1]
        basis = np.cos(np.pi / n * (np.arange(n) + 0.5)[:, None] * np.arange(count)[None, :])
        basis[:, 0] *= np.sqrt(1 / n)
        if count > 1:
            basis[:, 1:] *= np.sqrt(2 / n)
        return values @ basis


def mfcc(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 26,
    frame_length: int = 400,
    hop_length: int = 160,
    n_fft: int | None = None,
    preemphasis: float = 0.97,
    include_c0: bool = True,
) -> np.ndarray:
    """Compute frame-wise MFCCs from mono audio without requiring librosa."""
    if n_mfcc <= 0:
        raise ValueError("n_mfcc must be positive")
    x = pre_emphasis(signal, preemphasis)
    frames = frame_signal(x, frame_length, hop_length)
    if frames.shape[0] == 0:
        return np.empty((0, n_mfcc), dtype=np.float64)
    frames = window_frames(frames, "hamming")
    fft_size = max(frame_length, n_fft or frame_length)
    spectrum = np.abs(np.fft.rfft(frames, n=fft_size, axis=1)) ** 2 / fft_size
    energies = spectrum @ mel_filterbank(sample_rate, fft_size, n_mels).T
    cepstra = _dct_type_2(np.log(np.maximum(energies, np.finfo(float).tiny)), n_mfcc)
    if not include_c0 and cepstra.shape[1] > 0:
        cepstra[:, 0] = 0.0
    return cepstra


def delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Estimate time derivatives using the standard symmetric regression."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must have shape (frames, coefficients)")
    if width <= 0:
        raise ValueError("width must be positive")
    if x.shape[0] == 0:
        return x.copy()
    padded = np.pad(x, ((width, width), (0, 0)), mode="edge")
    denominator = 2 * sum(i * i for i in range(1, width + 1))
    result = np.zeros_like(x)
    for i in range(1, width + 1):
        result += i * (padded[width + i:width + i + len(x)] -
                       padded[width - i:width - i + len(x)])
    return result / denominator


def extract_features(
    signal: np.ndarray, sample_rate: int, include_deltas: bool = False, **kwargs: object
) -> np.ndarray:
    """Compute MFCCs, optionally appending first- and second-order deltas."""
    base = mfcc(signal, sample_rate, **kwargs)
    if not include_deltas or base.shape[0] == 0:
        return base
    first = delta(base)
    return np.concatenate((base, first, delta(first)), axis=1)
