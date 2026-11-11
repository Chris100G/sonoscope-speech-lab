"""Objective, frame-based speaking-practice measures built on NumPy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from sonoscope.dsp.audio import to_mono
from sonoscope.dsp.features import frame_signal
from sonoscope.dsp.speech import estimate_pitch


@dataclass(frozen=True)
class PracticeReport:
    """Interpretable timing and signal measures for one practice recording."""

    sample_rate: int
    duration_seconds: float
    speaking_seconds: float
    pause_seconds: float
    pause_count: int
    long_pause_count: int
    speech_segments: int
    word_count: int
    words_per_minute: float | None
    median_pitch_hz: float | None
    pitch_range_semitones: float | None
    average_level_dbfs: float
    peak_level_dbfs: float
    clipped_percent: float
    active_percent: float
    frame_times: tuple[float, ...]
    frame_levels_dbfs: tuple[float, ...]
    speech_mask: tuple[bool, ...]
    pitch_times: tuple[float, ...]
    pitch_values_hz: tuple[float | None, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly report fields and arrays."""
        return asdict(self)


def _speech_regions(mask: np.ndarray, minimum_gap: int) -> list[tuple[int, int]]:
    """Join brief VAD gaps and return half-open active-frame ranges."""
    joined = np.asarray(mask, dtype=bool).copy()
    if joined.size == 0:
        return []
    gaps = np.flatnonzero(np.diff(np.r_[False, joined, False].astype(np.int8)) == -1)
    starts = np.flatnonzero(np.diff(np.r_[False, joined, False].astype(np.int8)) == 1)
    for previous_end, next_start in zip(gaps[:-1], starts[1:], strict=True):
        if next_start - previous_end <= minimum_gap:
            joined[previous_end:next_start] = True
    transitions = np.diff(np.r_[False, joined, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def analyze_speech(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    transcript: str = "",
    *,
    silence_floor_dbfs: float = -48.0,
    minimum_pause_seconds: float = 0.25,
    long_pause_seconds: float = 1.0,
    frame_seconds: float = 0.025,
    hop_seconds: float = 0.010,
    pitch_stride: int = 8,
) -> PracticeReport:
    """Summarize speech timing and acoustics without judging the speaker.

    Word pace is calculated only when the user supplies an edited transcript.
    All thresholds are engineering heuristics, not medical or speech diagnoses.
    """
    signal = to_mono(np.asarray(audio, dtype=np.float64)).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not signal.size:
        raise ValueError("audio must contain at least one sample")
    if not np.all(np.isfinite(signal)):
        raise ValueError("audio must contain only finite samples")
    if not 0 <= minimum_pause_seconds < long_pause_seconds:
        raise ValueError("pause thresholds must be non-negative and ordered")
    if pitch_stride <= 0 or frame_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("frame, hop, and pitch stride must be positive")

    signal = np.clip(signal, -1.0, 1.0)
    frame_length = max(32, round(frame_seconds * sample_rate))
    hop_length = max(1, round(hop_seconds * sample_rate))
    frames = frame_signal(signal, frame_length, hop_length)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    levels = 20 * np.log10(np.maximum(rms, 1e-12))
    peak_dbfs = float(20 * np.log10(max(float(np.max(np.abs(signal))), 1e-12)))
    # Relative threshold helps accommodate quiet recordings; an absolute floor
    # prevents low-level microphone noise from counting as speech.
    threshold_dbfs = max(silence_floor_dbfs, peak_dbfs - 32.0)
    raw_mask = (
        levels >= threshold_dbfs
        if peak_dbfs >= silence_floor_dbfs
        else np.zeros_like(levels, dtype=bool)
    )
    minimum_gap = max(0, round(0.12 * sample_rate / hop_length))
    regions = _speech_regions(raw_mask, minimum_gap)
    mask = np.zeros_like(raw_mask, dtype=bool)
    for start, end in regions:
        mask[start:end] = True

    duration = len(signal) / sample_rate
    speaking_seconds = min(duration, float(mask.sum() * hop_length / sample_rate))
    pause_seconds = max(0.0, duration - speaking_seconds)
    pause_frames = [
        (next_start - end) * hop_length / sample_rate
        for (_, end), (next_start, _) in pairwise(regions)
        if (next_start - end) * hop_length / sample_rate >= minimum_pause_seconds
    ]
    words = sum(
        1 for token in transcript.split() if any(character.isalnum() for character in token)
    )
    wpm = 60.0 * words / speaking_seconds if words and speaking_seconds > 0 else None
    voiced_indices = np.flatnonzero(mask)[::pitch_stride]
    pitch_values: list[float | None] = []
    pitch_times: list[float] = []
    pitch_window = max(frame_length, round(0.04 * sample_rate))
    for index in voiced_indices:
        center = int(index * hop_length + frame_length / 2)
        start = max(0, min(len(signal) - pitch_window, center - pitch_window // 2))
        segment = signal[start : start + pitch_window]
        pitch = estimate_pitch(segment, sample_rate) if np.sqrt(np.mean(segment**2)) > 1e-5 else 0.0
        pitch_times.append(center / sample_rate)
        pitch_values.append(float(pitch) if pitch > 0 else None)
    voiced_pitch = np.asarray([value for value in pitch_values if value is not None], dtype=float)
    median_pitch = float(np.median(voiced_pitch)) if voiced_pitch.size else None
    pitch_range = (
        float(12 * np.log2(np.percentile(voiced_pitch, 75) / np.percentile(voiced_pitch, 25)))
        if voiced_pitch.size >= 4 and np.percentile(voiced_pitch, 25) > 0
        else None
    )
    active_rms = rms[mask]
    average_level = (
        float(20 * np.log10(max(float(np.sqrt(np.mean(active_rms**2))), 1e-12)))
        if active_rms.size
        else float(20 * np.log10(max(float(np.sqrt(np.mean(signal**2))), 1e-12)))
    )
    clipped = float(np.mean(np.abs(signal) >= 0.999) * 100)
    frame_times = (np.arange(len(frames)) * hop_length + frame_length / 2) / sample_rate
    return PracticeReport(
        sample_rate=sample_rate,
        duration_seconds=duration,
        speaking_seconds=speaking_seconds,
        pause_seconds=pause_seconds,
        pause_count=len(pause_frames),
        long_pause_count=sum(
            value >= long_pause_seconds - frame_seconds for value in pause_frames
        ),
        speech_segments=len(regions),
        word_count=words,
        words_per_minute=wpm,
        median_pitch_hz=median_pitch,
        pitch_range_semitones=pitch_range,
        average_level_dbfs=average_level,
        peak_level_dbfs=peak_dbfs,
        clipped_percent=clipped,
        active_percent=float(mask.mean() * 100) if mask.size else 0.0,
        frame_times=tuple(map(float, frame_times)),
        frame_levels_dbfs=tuple(map(float, levels)),
        speech_mask=tuple(map(bool, mask)),
        pitch_times=tuple(pitch_times),
        pitch_values_hz=tuple(pitch_values),
    )
