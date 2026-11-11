"""Tests for the SpeakReady objective speaking-practice measures."""

import numpy as np
import pytest

from sonoscope.coach import analyze_speech, compress_audio


def test_silence_reports_no_speech_or_pitch() -> None:
    report = analyze_speech(np.zeros(16_000), 16_000, "one two")
    assert report.speaking_seconds == 0
    assert report.words_per_minute is None
    assert report.median_pitch_hz is None
    assert report.pause_count == 0


def test_tone_bursts_report_speech_pauses_and_pitch() -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 200 * t)
    signal = np.r_[tone, np.zeros(sample_rate), tone]
    report = analyze_speech(signal, sample_rate, "one two three four")
    assert report.duration_seconds == pytest.approx(3.0, abs=0.01)
    assert report.speech_segments == 2
    assert report.pause_count == 1
    assert report.long_pause_count == 1
    assert report.median_pitch_hz == pytest.approx(200, abs=15)
    assert report.words_per_minute is not None
    assert report.words_per_minute > 0


def test_report_handles_multilingual_transcript_and_clipping() -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate) / sample_rate
    signal = 1.2 * np.sin(2 * np.pi * 180 * t)
    report = analyze_speech(signal, sample_rate, "नमस्ते ಕನ್ನಡ hello")
    assert report.word_count == 3
    assert report.clipped_percent > 0
    assert report.to_dict()["sample_rate"] == sample_rate


def test_quiet_recording_is_treated_as_silence_not_speech() -> None:
    report = analyze_speech(np.full(16_000, 1e-4), 16_000, "one two")
    assert report.active_percent == 0
    assert report.words_per_minute is None


def test_rejects_empty_and_invalid_input() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        analyze_speech(np.array([]))
    with pytest.raises(ValueError, match="sample_rate"):
        analyze_speech(np.ones(100), 0)


def test_compressor_preserves_silence_and_reduces_loud_tone() -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate) / sample_rate
    signal = 0.8 * np.sin(2 * np.pi * 220 * t)
    result = compress_audio(signal, sample_rate, threshold_db=-24, ratio=4, makeup_db=0)
    assert result.audio.shape == signal.shape
    assert result.output_peak_dbfs < result.input_peak_dbfs
    assert result.mean_reduction_db > 0
    silence = compress_audio(np.zeros(1000), sample_rate)
    assert np.all(silence.audio == 0)


def test_compressor_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="ratio"):
        compress_audio(np.ones(100), 16_000, ratio=0.5)
