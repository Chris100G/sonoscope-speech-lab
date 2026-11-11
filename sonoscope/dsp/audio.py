"""Audio loading and sample-rate conversion helpers."""

from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path

import numpy as np

AudioInput = str | Path | bytes | bytearray | memoryview


def _normalize_pcm(samples: np.ndarray) -> np.ndarray:
    """Convert integer PCM to float audio in approximately [-1, 1]."""
    data = np.asarray(samples)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        return data.astype(np.float64) / scale
    return data.astype(np.float64, copy=False)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert mono or channel-last multichannel audio to a mono float array."""
    data = _normalize_pcm(np.asarray(audio))
    if data.ndim == 1:
        return data
    if data.ndim != 2:
        raise ValueError("audio must be a one- or two-dimensional array")
    # WAV readers expose samples as (frames, channels); retain that convention.
    return np.mean(data, axis=1)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio, preferring SciPy polyphase filtering when available."""
    data = np.asarray(audio, dtype=np.float64)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or data.size == 0:
        return data.copy()
    if data.ndim not in (1, 2):
        raise ValueError("audio must be one- or two-dimensional")
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(source_rate), int(target_rate))
        return resample_poly(data, target_rate // divisor, source_rate // divisor, axis=0)
    except ImportError:
        # Linear interpolation keeps the backend usable without SciPy.
        n_out = max(1, round(data.shape[0] * target_rate / source_rate))
        old = np.arange(data.shape[0], dtype=np.float64)
        new = np.linspace(0, max(0, data.shape[0] - 1), n_out)
        if data.ndim == 1:
            return np.interp(new, old, data)
        return np.column_stack([np.interp(new, old, data[:, c]) for c in range(data.shape[1])])


def _read_wave_fallback(path: AudioInput) -> tuple[int, np.ndarray]:
    """Read common uncompressed PCM WAV encodings with the standard library."""
    source = BytesIO(bytes(path)) if isinstance(path, (bytes, bytearray, memoryview)) else str(path)
    with wave.open(source, "rb") as wav:
        channels, width, rate, count = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        raw = wav.readframes(count)
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2")
        data = _normalize_pcm(data)
    elif width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (octets[:, 0].astype(np.int32) |
                  (octets[:, 1].astype(np.int32) << 8) |
                  (octets[:, 2].astype(np.int32) << 16))
        values = (values ^ 0x800000) - 0x800000
        data = values.astype(np.float64) / 8388608.0
    elif width == 4:
        data = _normalize_pcm(np.frombuffer(raw, dtype="<i4"))
    else:
        raise ValueError(f"unsupported PCM sample width: {width} bytes")
    return rate, data.reshape(-1, channels) if channels > 1 else data


def load_audio(
    path: AudioInput, target_sr: int | None = 16000, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Load WAV, FLAC, or supported MP3 audio as normalized float samples.

    Returns ``(audio, sample_rate)``. SoundFile handles the supported formats;
    a standard-library WAV fallback is retained for minimal installations.
    """
    try:
        import soundfile as sf
    except ImportError:
        rate, audio = _read_wave_fallback(path)
    else:
        source = BytesIO(bytes(path)) if isinstance(path, (bytes, bytearray, memoryview)) else str(path)
        audio, rate = sf.read(source, dtype="float64", always_2d=False)
    if mono:
        audio = to_mono(audio)
    if target_sr is not None and target_sr != rate:
        audio = resample_audio(audio, int(rate), int(target_sr))
        rate = int(target_sr)
    return np.asarray(audio, dtype=np.float64), int(rate)
