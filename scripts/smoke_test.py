"""Headless Streamlit application smoke test."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Load the complete app once and fail on any uncaught Streamlit exception."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Streamlit smoke test failed:\n{details}")
    expected_tabs = {"Practice", "My progress", "Audio compressor", "How it works"}
    rendered_tabs = {tab.label for tab in app.tabs}
    if not expected_tabs.issubset(rendered_tabs):
        raise SystemExit(f"Navigation tabs missing: {sorted(expected_tabs - rendered_tabs)}")
    if not any(button.label == "Compress and preview" for button in app.button):
        raise SystemExit("The audio compressor workflow was not rendered.")
    example_toggle = next((toggle for toggle in app.toggle if toggle.label == "Show a clearly illustrative example"), None)
    if example_toggle is None:
        raise SystemExit("The illustrative practice example was not rendered.")
    sample_rate = 16_000
    time_axis = np.arange(sample_rate) / sample_rate
    tone = 0.18 * np.sin(2 * np.pi * 190 * time_axis)
    sample_audio = np.r_[tone, np.zeros(sample_rate // 3), tone]
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, sample_audio, sample_rate, format="WAV")
    app.file_uploader[0].set_value(("practice.wav", wav_buffer.getvalue(), "audio/wav"))
    app.run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Audio upload smoke test failed:\n{details}")
    transcript = next((field for field in app.text_area if field.label == "Your transcript (optional)"), None)
    if transcript is None:
        raise SystemExit("Practice transcript control was not rendered.")
    transcript.set_value("This is an example answer.")
    app.run()
    review_button = next((button for button in app.button if button.label == "Review this take"), None)
    if review_button is None:
        raise SystemExit("Practice review control did not appear after upload.")
    review_button.click()
    app.run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Practice analysis smoke test failed:\n{details}")
    if len(app.get("plotly_chart")) < 3:
        raise SystemExit("Practice waveform, rhythm and pitch charts were not rendered.")
    if not any(metric.label == "Speaking pace" for metric in app.metric):
        raise SystemExit("Speaking pace report was not rendered.")

    app.file_uploader[1].set_value(("compress.wav", wav_buffer.getvalue(), "audio/wav"))
    app.run()
    compressor_button = next((button for button in app.button if button.label == "Compress and preview"), None)
    if compressor_button is None or compressor_button.disabled:
        raise SystemExit("Compressor did not enable after audio upload.")
    compressor_button.click()
    app.run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Compressor smoke test failed:\n{details}")
    if not any(metric.label == "Average gain reduction" for metric in app.metric):
        raise SystemExit("Compressor output report was not rendered.")
    print("SpeakReady app, audio analysis, charts, and compressor workflows passed.")


if __name__ == "__main__":
    main()
