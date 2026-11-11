"""Transparent mono dynamics compression using a soft-knee RMS detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompressionResult:
    """Processed samples and the gain-reduction curve used to create them."""

    audio: np.ndarray
    gain_db: np.ndarray
    input_peak_dbfs: float
    output_peak_dbfs: float
    mean_reduction_db: float


def _soft_knee_gain(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Compute static compression gain in decibels."""
    overshoot = level_db - threshold_db
    if knee_db <= 0:
        output_level = np.where(
            overshoot > 0,
            threshold_db + overshoot / ratio,
            level_db,
        )
    else:
        half_knee = knee_db / 2
        upper = overshoot >= half_knee
        lower = overshoot <= -half_knee
        in_knee = ~(upper | lower)
        output_level = level_db.copy()
        output_level[upper] = threshold_db + overshoot[upper] / ratio
        output_level[in_knee] += (
            (1 / ratio - 1)
            * (overshoot[in_knee] + half_knee) ** 2
            / (2 * knee_db)
        )
    return output_level - level_db


def compress_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -24.0,
    ratio: float = 3.0,
    knee_db: float = 6.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    makeup_db: float = 0.0,
    detector_hop_ms: float = 5.0,
) -> CompressionResult:
    """Compress mono speech with an RMS detector and attack/release envelope.

    This is a fixed-threshold feed-forward compressor, not a loudness meter or
    brick-wall peak limiter. Audio is never written to disk by this function.
    """
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError("audio must contain at least one sample")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(signal)):
        raise ValueError("audio must contain only finite samples")
    if not 1 <= ratio <= 20:
        raise ValueError("ratio must be between 1:1 and 20:1")
    if not 0 <= knee_db <= 24:
        raise ValueError("knee must be between 0 and 24 dB")
    if attack_ms <= 0 or release_ms <= 0 or detector_hop_ms <= 0:
        raise ValueError("attack, release, and detector hop must be positive")
    if not -80 <= threshold_db <= 0 or not -12 <= makeup_db <= 24:
        raise ValueError("threshold or makeup gain is outside the supported range")

    hop = max(1, round(detector_hop_ms * sample_rate / 1000))
    count = int(np.ceil(len(signal) / hop))
    padded = np.pad(signal, (0, count * hop - len(signal)))
    blocks = padded.reshape(count, hop)
    powers = np.mean(blocks * blocks, axis=1)
    envelope = np.empty(count, dtype=np.float64)
    current = 0.0
    attack = np.exp(-detector_hop_ms / attack_ms)
    release = np.exp(-detector_hop_ms / release_ms)
    for index, power in enumerate(powers):
        coefficient = attack if power > current else release
        current = coefficient * current + (1 - coefficient) * power
        envelope[index] = current

    level_db = 10 * np.log10(np.maximum(envelope, 1e-12))
    block_gain_db = _soft_knee_gain(level_db, threshold_db, ratio, knee_db) + makeup_db
    sample_positions = np.arange(len(signal), dtype=float) / hop
    gain_db = np.interp(sample_positions, np.arange(count), block_gain_db)
    compressed = signal * np.power(10.0, gain_db / 20)
    input_peak = float(20 * np.log10(max(float(np.max(np.abs(signal))), 1e-12)))
    output_peak = float(20 * np.log10(max(float(np.max(np.abs(compressed))), 1e-12)))
    reduction = np.maximum(0.0, -block_gain_db + makeup_db)
    active = envelope > 1e-10
    mean_reduction = float(np.mean(reduction[active])) if np.any(active) else 0.0
    return CompressionResult(compressed, gain_db, input_peak, output_peak, mean_reduction)
