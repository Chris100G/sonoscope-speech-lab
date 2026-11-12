"""Session management, persistence, and export/import for SpeakReady."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import streamlit as st

SESSION_KEY = "speakready_session"


def _new_session() -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": 2,
        "takes": [],
        "notes": [],
        "settings": {
            "theme": "light",
            "default_prompt": "Interview · tell me about a challenge you solved",
            "default_target": "Steady conversation",
            "custom_pace_min": 110,
            "custom_pace_max": 160,
            "dark_mode": False,
        },
        "metadata": {
            "app": "SpeakReady",
            "purpose": "speaking-practice",
        },
    }


def ensure_session() -> dict:
    """Guarantee a session object exists in Streamlit state."""
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = _new_session()
    return st.session_state[SESSION_KEY]


def reset_session() -> None:
    """Clear the persisted session back to defaults."""
    st.session_state[SESSION_KEY] = _new_session()


def add_take(session: dict, report: dict, target: tuple[int, int], prompt_name: str, prompt_text: str, transcript: str) -> None:
    """Append a saved practice take to the session."""
    take = {
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_seconds": report["duration_seconds"],
        "speaking_seconds": report["speaking_seconds"],
        "pause_seconds": report["pause_seconds"],
        "pause_count": report["pause_count"],
        "long_pause_count": report["long_pause_count"],
        "speech_segments": report["speech_segments"],
        "word_count": report["word_count"],
        "words_per_minute": report["words_per_minute"],
        "median_pitch_hz": report["median_pitch_hz"],
        "pitch_range_semitones": report["pitch_range_semitones"],
        "average_level_dbfs": report["average_level_dbfs"],
        "peak_level_dbfs": report["peak_level_dbfs"],
        "clipped_percent": report["clipped_percent"],
        "active_percent": report["active_percent"],
        "target_min": target[0],
        "target_max": target[1],
        "target_name": prompt_name,
        "prompt": prompt_name,
        "prompt_text": prompt_text,
        "transcript": transcript,
        "audio_digest": hashlib.sha256(report.get("audio_bytes", b"").encode() if isinstance(report.get("audio_bytes"), str) else b"").hexdigest()[:16],
    }
    session["takes"].append(take)


TARGET_NAMES = {"Steady conversation", "Interview answer", "Presentation", "My own range"}


def add_note(session: dict, text: str, take_index: int | None = None) -> None:
    """Append a free-form journal note tied to an optional take."""
    session["notes"].append(
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "text": text,
            "take_index": take_index,
        }
    )


def export_session(session: dict) -> bytes:
    """Serialize the full session to JSON bytes for download."""
    return json.dumps(session, ensure_ascii=False, indent=2).encode("utf-8")


def import_session(data: bytes) -> dict:
    """Validate and return an imported session object."""
    raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 2:
        raise ValueError("Unsupported session export format.")
    return raw
