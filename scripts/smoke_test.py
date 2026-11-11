"""Headless Streamlit application smoke test."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Load the complete app once and fail on any uncaught Streamlit exception."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Streamlit smoke test failed:\n{details}")
    expected_tabs = {
        "Home",
        "Signal X-Ray",
        "Vocal Tract Studio",
        "Distance Lab",
        "HMM Lab",
        "Voice Assistant",
        "Benchmark & Ethics",
        "Pronunciation Coach",
    }
    rendered_tabs = {tab.label for tab in app.tabs}
    if not expected_tabs.issubset(rendered_tabs):
        raise SystemExit(f"Navigation tabs missing: {sorted(expected_tabs - rendered_tabs)}")
    if len(app.get("plotly_chart")) < 3:
        raise SystemExit("Expected homepage signal and recognition plots were not rendered.")
    demo_button = next((button for button in app.button if button.label == "Run demo recognition"), None)
    if demo_button is None:
        raise SystemExit("Synthetic demo recognition control was not rendered.")
    demo_button.click()
    app.run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Demo recognition smoke test failed:\n{details}")
    if not any("Matched" in str(message.value) for message in app.success):
        raise SystemExit("Synthetic demo recognition did not return a match.")
    assistant_button = next(
        (button for button in app.button if button.label == "Recognize with DTW + VQ-HMM"),
        None,
    )
    if assistant_button is None:
        raise SystemExit("DTW + VQ-HMM recognition control was not rendered.")
    assistant_button.click()
    app.run()
    if app.exception:
        details = "\n".join(str(error) for error in app.exception)
        raise SystemExit(f"Assistant recognition smoke test failed:\n{details}")
    print("SONOSCOPE Streamlit app loaded successfully.")


if __name__ == "__main__":
    main()
