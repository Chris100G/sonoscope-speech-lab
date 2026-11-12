"""Tests for SpeakReady session management and charts."""

import json

import pytest

from sonoscope.ui.session import (
    SESSION_KEY,
    add_note,
    add_take,
    ensure_session,
    export_session,
    import_session,
    reset_session,
)


def _fake_session_state(monkeypatch):
    state = {}

    class FakeState(dict):
        def __setattr__(self, key, value):
            self[key] = value

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

        def pop(self, key, *args):
            return dict.pop(self, key, *args)

        def setdefault(self, key, default):
            return dict.setdefault(self, key, default)

    fake = FakeState(state)
    monkeypatch.setattr("sonoscope.ui.session.st.session_state", fake)
    return fake


def test_ensure_session_creates_default(monkeypatch):
    store = _fake_session_state(monkeypatch)
    session = ensure_session()
    assert session["version"] == 2
    assert session["takes"] == []
    assert session["notes"] == []
    assert SESSION_KEY in store


def test_reset_session_clears_takes(monkeypatch):
    _fake_session_state(monkeypatch)
    session = ensure_session()
    add_take(
        session,
        {
            "duration_seconds": 5.0,
            "speaking_seconds": 3.0,
            "pause_seconds": 2.0,
            "pause_count": 1,
            "long_pause_count": 0,
            "speech_segments": 1,
            "word_count": 10,
            "words_per_minute": 120.0,
            "median_pitch_hz": 150.0,
            "pitch_range_semitones": 4.0,
            "average_level_dbfs": -20.0,
            "peak_level_dbfs": -5.0,
            "clipped_percent": 0.0,
            "active_percent": 60.0,
            "frame_times": (),
            "frame_levels_dbfs": (),
            "speech_mask": (),
            "pitch_times": (),
            "pitch_values_hz": (),
        },
        (110, 160),
        "Interview · tell me about a challenge you solved",
        "Tell me about a challenge you solved and what you learned.",
        "I solved a challenge.",
    )
    assert len(session["takes"]) == 1
    reset_session()
    session = ensure_session()
    assert session["takes"] == []


def test_add_take_records_fields(monkeypatch):
    _fake_session_state(monkeypatch)
    session = ensure_session()
    report = {
        "duration_seconds": 6.0,
        "speaking_seconds": 4.0,
        "pause_seconds": 2.0,
        "pause_count": 2,
        "long_pause_count": 1,
        "speech_segments": 2,
        "word_count": 20,
        "words_per_minute": 150.0,
        "median_pitch_hz": 140.0,
        "pitch_range_semitones": 5.0,
        "average_level_dbfs": -22.0,
        "peak_level_dbfs": -3.0,
        "clipped_percent": 0.5,
        "active_percent": 66.0,
        "frame_times": (),
        "frame_levels_dbfs": (),
        "speech_mask": (),
        "pitch_times": (),
        "pitch_values_hz": (),
    }
    add_take(session, report, (105, 155), "Presentation · explain one idea simply", "Explain an idea.", "Hello world.")
    take = session["takes"][0]
    assert take["words_per_minute"] == 150.0
    assert take["pause_count"] == 2
    assert take["long_pause_count"] == 1
    assert take["target_name"] == "Presentation · explain one idea simply"
    assert take["prompt"] == "Presentation · explain one idea simply"
    assert take["prompt_text"] == "Explain an idea."
    assert take["transcript"] == "Hello world."
    assert "saved_at" in take


def test_add_note_tracks_take_index(monkeypatch):
    _fake_session_state(monkeypatch)
    session = ensure_session()
    add_note(session, "Great pacing.", take_index=0)
    assert len(session["notes"]) == 1
    assert session["notes"][0]["text"] == "Great pacing."
    assert session["notes"][0]["take_index"] == 0
    assert "created_at" in session["notes"][0]


def test_export_import_roundtrip(monkeypatch):
    _fake_session_state(monkeypatch)
    session = ensure_session()
    add_take(
        session,
        {
            "duration_seconds": 5.0,
            "speaking_seconds": 3.0,
            "pause_seconds": 2.0,
            "pause_count": 1,
            "long_pause_count": 0,
            "speech_segments": 1,
            "word_count": 8,
            "words_per_minute": 100.0,
            "median_pitch_hz": 130.0,
            "pitch_range_semitones": 3.0,
            "average_level_dbfs": -25.0,
            "peak_level_dbfs": -4.0,
            "clipped_percent": 0.0,
            "active_percent": 60.0,
            "frame_times": (),
            "frame_levels_dbfs": (),
            "speech_mask": (),
            "pitch_times": (),
            "pitch_values_hz": (),
        },
        (110, 160),
        "Meeting · give a concise project update",
        "Status update.",
        "The project is on track.",
    )
    data = export_session(session)
    imported = import_session(data)
    assert imported["version"] == 2
    assert len(imported["takes"]) == 1
    assert imported["takes"][0]["words_per_minute"] == 100.0
    assert imported["created_at"] == session["created_at"]


def test_import_rejects_bad_version():
    with pytest.raises(ValueError, match="Unsupported session export format"):
        import_session(json.dumps({"version": 1}).encode("utf-8"))
