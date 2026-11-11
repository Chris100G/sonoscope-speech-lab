"""Small multilingual command demo with entirely synthetic audio examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonoscope.dsp.features import mfcc
from sonoscope.recognizers import DTWTemplateRecognizer, RecognitionResult


@dataclass(frozen=True)
class DemoCommand:
    """A demo phrase with language, intent, and synthetic tone pattern."""

    key: str
    text: str
    language: str
    intent: str
    pattern: tuple[float, ...]


def demo_commands() -> tuple[DemoCommand, ...]:
    """Return the built-in English, Hindi, and Kannada command catalogue."""
    return (
        DemoCommand("en_start", "Start recording", "en", "start", (4, 7, 5, 9)),
        DemoCommand("hi_start", "रिकॉर्डिंग शुरू करो", "hi", "start", (4, 8, 5, 10)),
        DemoCommand("kn_start", "ರೆಕಾರ್ಡಿಂಗ್ ಪ್ರಾರಂಭಿಸಿ", "kn", "start", (3, 7, 6, 9)),
        DemoCommand("en_stop", "Stop recording", "en", "stop", (10, 5, 8, 4)),
        DemoCommand("hi_stop", "रिकॉर्डिंग बंद करो", "hi", "stop", (10, 6, 8, 3)),
        DemoCommand("kn_stop", "ರೆಕಾರ್ಡಿಂಗ್ ನಿಲ್ಲಿಸಿ", "kn", "stop", (9, 5, 7, 3)),
        DemoCommand("en_save", "Save recording", "en", "save", (5, 10, 4, 8, 6)),
        DemoCommand("hi_save", "रिकॉर्डिंग सहेजो", "hi", "save", (5, 9, 4, 8, 7)),
        DemoCommand("kn_save", "ರೆಕಾರ್ಡಿಂಗ್ ಉಳಿಸಿ", "kn", "save", (6, 10, 4, 9)),
        DemoCommand("en_todo_add", "Add a task", "en", "todo_add", (2, 11, 6, 3)),
        DemoCommand("hi_todo_add", "एक काम जोड़ो", "hi", "todo_add", (2, 12, 6, 4)),
        DemoCommand("kn_todo_add", "ಒಂದು ಕೆಲಸ ಸೇರಿಸಿ", "kn", "todo_add", (1, 11, 7, 3)),
        DemoCommand("en_todo_done", "Complete a task", "en", "todo_done", (11, 2, 9, 6)),
        DemoCommand("hi_todo_done", "काम पूरा हुआ", "hi", "todo_done", (12, 2, 9, 7)),
        DemoCommand("kn_todo_done", "ಕೆಲಸ ಮುಗಿದಿದೆ", "kn", "todo_done", (11, 1, 8, 6)),
    )


def synthesize_command(
    command: DemoCommand | str, sample_rate: int = 16000, segment_seconds: float = 0.11
) -> np.ndarray:
    """Generate a deterministic tone-pattern utterance (not speech audio)."""
    if isinstance(command, str):
        matches = [item for item in demo_commands() if item.key == command]
        if not matches:
            raise KeyError(command)
        command = matches[0]
    if sample_rate <= 0 or segment_seconds <= 0:
        raise ValueError("sample_rate and segment_seconds must be positive")
    count = max(8, int(sample_rate * segment_seconds))
    pieces = []
    for index, tone in enumerate(command.pattern):
        frequency = 180.0 + 29.0 * tone
        t = np.arange(count, dtype=float) / sample_rate
        envelope = np.sin(np.linspace(0, np.pi, count)) ** 2
        piece = envelope * (
            np.sin(2 * np.pi * frequency * t) +
            0.25 * np.sin(2 * np.pi * frequency * 2.03 * t)
        )
        pieces.append(piece / (1 + index * 0.03))
    return np.concatenate(pieces).astype(np.float64)


def build_demo_recognizer(sample_rate: int = 16000) -> DTWTemplateRecognizer:
    """Build a DTW classifier from MFCCs of the built-in synthetic patterns."""
    templates = {
        command.key: mfcc(synthesize_command(command, sample_rate), sample_rate)
        for command in demo_commands()
    }
    return DTWTemplateRecognizer(reject_threshold=5.0).fit(templates)


def recognize_demo_command(
    audio: np.ndarray,
    sample_rate: int = 16000,
    recognizer: DTWTemplateRecognizer | None = None,
    reject_threshold: float | None = None,
) -> RecognitionResult:
    """Recognize a synthetic demo command waveform and return its result."""
    model = recognizer or build_demo_recognizer(sample_rate)
    features = mfcc(audio, sample_rate)
    return model.recognize(features, reject_threshold)
