"""SpeakReady: a practical, private speaking-practice and audio-toolkit app."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from sonoscope.coach import analyze_speech, compress_audio
from sonoscope.dsp.audio import load_audio
from sonoscope.ui.style import apply_styles

SAMPLE_RATE = 16_000
TARGETS = {
    "Steady conversation": (110, 160),
    "Interview answer": (115, 165),
    "Presentation": (105, 155),
    "My own range": None,
}

st.set_page_config(
    page_title="SpeakReady | Speaking practice",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()


def chart(figure: go.Figure, key: str) -> None:
    """Render a responsive chart with an explicit unique element key."""
    st.plotly_chart(figure, width="stretch", key=key)


def wave_chart(signal: np.ndarray, sample_rate: int, title: str, key: str) -> go.Figure:
    step = max(1, len(signal) // 12_000)
    figure = go.Figure(
        go.Scattergl(
            x=np.arange(0, len(signal), step) / sample_rate,
            y=signal[::step],
            mode="lines",
            line={"color": "#167c80", "width": 1.25},
            fill="tozeroy",
            fillcolor="rgba(22,124,128,.1)",
            name="audio",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.02},
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        margin={"l": 25, "r": 15, "t": 55, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(24,62,68,.08)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(24,62,68,.08)")
    return figure


def audio_controls(prefix: str) -> tuple[np.ndarray, int] | None:
    """Offer browser recording or audio upload and normalize to mono 16 kHz."""
    upload, microphone = st.columns(2)
    with upload:
        uploaded = st.file_uploader(
            "Choose WAV, FLAC, or MP3",
            type=["wav", "flac", "mp3"],
            key=f"{prefix}-upload",
            help="A 15–60 second clip is a useful practice length.",
        )
    with microphone:
        recording = st.audio_input("Record a practice take", key=f"{prefix}-mic")
    source = recording if recording is not None else uploaded
    if source is None:
        return None
    try:
        audio, rate = load_audio(source.getvalue(), target_sr=SAMPLE_RATE)
    except (ValueError, RuntimeError, sf.SoundFileRuntimeError) as error:
        st.error(f"Could not read that audio file: {error}")
        return None
    if not audio.size:
        st.error("This audio contains no samples.")
        return None
    if len(audio) / rate > 90:
        st.error("For a quick, responsive analysis, please choose a clip under 90 seconds.")
        return None
    st.caption(
        f"{'Microphone recording' if recording is not None else 'Uploaded audio'} "
        f"· {len(audio) / rate:.1f} s · {rate:,} Hz · mono"
    )
    return audio, rate


def pace_guidance(wpm: float | None, target: tuple[int, int]) -> tuple[str, str]:
    if wpm is None:
        return "Add your words", "Paste an edited transcript above to estimate speaking pace."
    low, high = target
    if wpm > high:
        return "Try a little more space", "Pause at commas and full stops. Keep the words that carry your point."
    if wpm < low:
        return "Try joining a phrase", "If the pauses felt unplanned, rehearse one sentence as a single thought."
    return "Your pace is in your chosen range", "Keep this take as a reference, then compare how a second take feels."


def timeline_chart(report: dict[str, object]) -> go.Figure:
    times = np.asarray(report["frame_times"])
    levels = np.asarray(report["frame_levels_dbfs"])
    speech = np.asarray(report["speech_mask"], dtype=bool)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=times,
            y=levels,
            mode="lines",
            line={"color": "#809497", "width": 1},
            name="Frame level",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times[speech],
            y=levels[speech],
            mode="markers",
            marker={"color": "#167c80", "size": 4},
            name="Detected speech frames",
        )
    )
    figure.update_layout(
        title={"text": "Speaking rhythm · detected speech and pauses", "x": .02},
        xaxis_title="Time (seconds)",
        yaxis_title="RMS level (dBFS)",
        height=340,
        margin={"l": 25, "r": 20, "t": 55, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def pitch_chart(report: dict[str, object]) -> go.Figure:
    pitches = np.asarray(
        [np.nan if value is None else value for value in report["pitch_values_hz"]],
        dtype=float,
    )
    figure = go.Figure(
        go.Scatter(
            x=report["pitch_times"],
            y=pitches,
            mode="lines+markers",
            connectgaps=False,
            marker={"size": 4, "color": "#cf8d48"},
            line={"color": "#cf8d48", "width": 1.5},
            name="Estimated pitch",
        )
    )
    figure.update_layout(
        title={"text": "Pitch movement · voiced frames only", "x": .02},
        xaxis_title="Time (seconds)",
        yaxis_title="Estimated pitch (Hz)",
        height=300,
        margin={"l": 25, "r": 20, "t": 55, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def render_practice() -> None:
    st.markdown('<div class="sono-section-label">Your next take, made more useful</div>', unsafe_allow_html=True)
    st.header("Practice a short answer")
    st.write("Record or upload a take, add your own transcript, and get a few concrete things to try next.")
    prompts = {
        "Interview · tell me about a challenge you solved": "Tell me about a challenge you solved and what you learned.",
        "Meeting · give a concise project update": "What is the current status, and what is one next step?",
        "Presentation · explain one idea simply": "What is one idea you want your audience to remember?",
        "Everyday · make a clear request": "What would help you, and why?",
    }
    prompt_name = st.selectbox("Choose a real-world rehearsal prompt", list(prompts), key="practice-prompt")
    st.markdown(f"**Try this:** {prompts[prompt_name]}")
    source = audio_controls("practice")
    if source is None:
        st.info("Start with a short answer to a meeting, interview, or presentation question. Your audio is analysed in memory, not saved in your practice history.")
        return
    audio, sample_rate = source
    st.audio(audio, sample_rate=sample_rate)
    st.plotly_chart(
        wave_chart(audio, sample_rate, "Your recording", "practice-waveform"),
        width="stretch",
        key="practice-wave",
    )
    transcript = st.text_area(
        "Your transcript (optional)",
        placeholder="Paste what you said. This helps estimate words per minute; it is not sent to a transcription service.",
        height=100,
        key="practice-transcript",
    )
    target_name = st.selectbox("Choose a pace reference", list(TARGETS), key="practice-target")
    if target_name == "My own range":
        low_col, high_col = st.columns(2)
        with low_col:
            low = st.number_input("Comfortable minimum (words/min)", 40, 260, 110, 5)
        with high_col:
            high = st.number_input("Comfortable maximum (words/min)", 50, 300, 160, 5)
        target = (int(low), int(high))
    else:
        target = TARGETS[target_name]
        assert target is not None
        st.caption(
            f"Reference only: {target[0]}–{target[1]} words/min. "
            "Comfortable pace varies by language, setting and speaking style."
        )
    if target[0] >= target[1]:
        st.error("Choose a maximum pace greater than the minimum.")
        return
    audio_digest = hashlib.sha256(audio.tobytes()).hexdigest()
    if st.button("Review this take", type="primary"):
        try:
            report = analyze_speech(audio, sample_rate, transcript)
        except ValueError as error:
            st.error(f"Could not analyse this take: {error}")
        else:
            st.session_state["speakready_report"] = report.to_dict()
            st.session_state["speakready_target"] = target
            st.session_state["speakready_transcript"] = transcript
            st.session_state["speakready_audio_digest"] = audio_digest
    report = st.session_state.get("speakready_report")
    if (
        report is not None
        and (
            st.session_state.get("speakready_audio_digest") != audio_digest
            or st.session_state.get("speakready_transcript") != transcript
            or st.session_state.get("speakready_target") != target
        )
    ):
        report = None
    if not report:
        st.caption("A transcript is optional. Without one, rhythm, pauses, pitch and recording level are still measured.")
        return
    target = st.session_state.get("speakready_target", target)
    title, tip = pace_guidance(report["words_per_minute"], target)
    st.divider()
    st.markdown('<div class="sono-section-label">Your practice notes</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.write(tip)
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Speaking pace",
        f"{report['words_per_minute']:.0f} wpm" if report["words_per_minute"] is not None else "Add transcript",
    )
    metric_cols[1].metric("Speaking time", f"{report['speaking_seconds']:.1f} s")
    metric_cols[2].metric("Detected pauses", str(report["pause_count"]))
    metric_cols[3].metric("Long pauses", str(report["long_pause_count"]))
    chart(timeline_chart(report), "practice-timeline")
    chart(pitch_chart(report), "practice-pitch")
    details, other = st.columns(2)
    with details:
        st.markdown('<div class="sono-section-label">Signal, not a score</div>', unsafe_allow_html=True)
        pitch_value = (
            f"{report['median_pitch_hz']:.0f} Hz median"
            if report["median_pitch_hz"] is not None
            else "Not enough voiced frames"
        )
        variation = (
            f"{report['pitch_range_semitones']:.1f} semitones (middle half of voiced frames)"
            if report["pitch_range_semitones"] is not None
            else "Not enough voiced frames"
        )
        st.markdown(
            f"- Pitch: **{pitch_value}**\n"
            f"- Pitch variation: **{variation}**\n"
            f"- Average voiced level: **{report['average_level_dbfs']:.1f} dBFS**\n"
            f"- Peak / clipped samples: **{report['peak_level_dbfs']:.1f} dBFS / {report['clipped_percent']:.2f}%**"
        )
    with other:
        st.markdown('<div class="sono-section-label">One small next step</div>', unsafe_allow_html=True)
        if report["long_pause_count"]:
            st.markdown(
                f"You had **{report['long_pause_count']} longer pause(s)**. Try using one intentional breath between your main points."
            )
        elif report["pause_count"] == 0:
            st.markdown("No longer internal pauses were detected. Try adding a breath at one natural transition.")
        else:
            st.markdown(
                f"You have **{report['pause_count']} natural pause(s)**. Keep the one that gives your main point room."
            )
        if report["clipped_percent"] > 0.1:
            st.warning("Some audio peaks are clipped. Move the microphone slightly farther away or lower its input level.")
        st.caption("Recording level depends on the device and distance; compare takes made with the same setup.")
    if st.button("Save these notes to this session's history"):
        history = st.session_state.setdefault("speakready_history", [])
        history.append(
            {
                "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_seconds": report["duration_seconds"],
                "speaking_seconds": report["speaking_seconds"],
                "word_count": report["word_count"],
                "words_per_minute": report["words_per_minute"],
                "pause_count": report["pause_count"],
                "pitch_range_semitones": report["pitch_range_semitones"],
                "target": target_name,
                "prompt": prompt_name,
            }
        )
        st.success("Notes saved in this browser session. The audio itself was not saved.")


def render_compressor() -> None:
    st.markdown('<div class="sono-section-label">Optional tool · no audio stored</div>', unsafe_allow_html=True)
    st.header("Make a recording easier to hear")
    st.write(
        "A gentle soft-knee compressor can bring quieter parts closer to louder parts. "
        "It does not remove room noise or restore already-clipped audio."
    )
    source = audio_controls("compressor")
    if source is None:
        st.info("Upload or record a take to create a processed listening copy.")
        audio, sample_rate = np.zeros(1, dtype=np.float64), SAMPLE_RATE
    else:
        audio, sample_rate = source
    settings1, settings2 = st.columns(2)
    with settings1:
        threshold = st.slider("Threshold", -48, -3, -24, 1, format="%d dBFS")
        ratio = st.slider("Ratio", 1.0, 10.0, 3.0, 0.5, format="%.1f:1")
        knee = st.slider("Soft knee", 0, 18, 6, 1, format="%d dB")
    with settings2:
        attack = st.slider("Attack", 1, 80, 10, 1, format="%d ms")
        release = st.slider("Release", 20, 500, 120, 10, format="%d ms")
        makeup = st.slider("Make-up gain", -6, 12, 0, 1, format="%d dB")
    if st.button("Compress and preview", type="primary", disabled=source is None):
        result = compress_audio(
            audio,
            sample_rate,
            threshold_db=threshold,
            ratio=ratio,
            knee_db=knee,
            attack_ms=attack,
            release_ms=release,
            makeup_db=makeup,
        )
        st.session_state["speakready_compressed"] = result.audio
        st.session_state["speakready_compression_report"] = {
            "input_peak": result.input_peak_dbfs,
            "output_peak": result.output_peak_dbfs,
            "mean_reduction": result.mean_reduction_db,
            "gain_db": result.gain_db,
            "sample_rate": sample_rate,
            "input_digest": hashlib.sha256(audio.tobytes()).hexdigest(),
        }
    output = st.session_state.get("speakready_compressed")
    report = st.session_state.get("speakready_compression_report")
    current_digest = hashlib.sha256(audio.tobytes()).hexdigest() if source is not None else None
    if report is not None and report.get("input_digest") != current_digest:
        output = None
        report = None
    if output is None or report is None:
        return
    metrics = st.columns(3)
    metrics[0].metric("Input peak", f"{report['input_peak']:.1f} dBFS")
    metrics[1].metric("Output peak", f"{report['output_peak']:.1f} dBFS")
    metrics[2].metric("Average gain reduction", f"{report['mean_reduction']:.1f} dB")
    before, after = st.columns(2)
    with before:
        chart(wave_chart(audio, sample_rate, "Original take", "compress-before"), "compress-before")
        st.audio(audio, sample_rate=sample_rate)
    with after:
        chart(wave_chart(output, sample_rate, "Compressed copy", "compress-after"), "compress-after")
        st.audio(output, sample_rate=sample_rate)
    figure = go.Figure(
        go.Scatter(
            x=np.arange(
                0,
                len(report["gain_db"]),
                max(1, len(report["gain_db"]) // 6_000),
            )
            / sample_rate,
            y=report["gain_db"][:: max(1, len(report["gain_db"]) // 6_000)],
            mode="lines",
            line={"color": "#cf8d48"},
            name="Applied gain",
        )
    )
    figure.update_layout(
        title="Compressor gain over time",
        xaxis_title="Time (seconds)",
        yaxis_title="Gain (dB)",
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    chart(figure, "compress-gain")
    buffer = io.BytesIO()
    sf.write(buffer, output, int(report["sample_rate"]), format="WAV", subtype="PCM_16")
    st.download_button(
        "Download processed WAV",
        data=buffer.getvalue(),
        file_name="speakready-compressed.wav",
        mime="audio/wav",
    )
    if report["output_peak"] > 0:
        st.warning("The processed peak exceeds 0 dBFS; lower make-up gain to avoid clipping on export.")
    st.caption("The compressor runs in memory. It is not a brick-wall limiter or a noise remover.")


def render_progress() -> None:
    st.markdown('<div class="sono-section-label">Practice is personal</div>', unsafe_allow_html=True)
    st.header("Your recent takes")
    st.write("Compare your own sessions, using the same recording setup when possible.")
    history = st.session_state.get("speakready_history", [])
    if not history:
        st.info("Save a practice note to start a private, session-only history.")
        demo = st.toggle("Show a clearly illustrative example", key="show-example")
        if demo:
            example = pd.DataFrame(
                {
                    "Take": ["Reference take", "Practice 1", "Practice 2"],
                    "Speaking pace (illustrative)": [172, 158, 146],
                    "Pauses (illustrative)": [1, 2, 3],
                }
            )
            st.caption("Illustration only—not recorded speech or measured user data.")
            from plotly.subplots import make_subplots

            figure = make_subplots(specs=[[{"secondary_y": True}]])
            figure.add_trace(
                go.Scatter(
                    x=example["Take"],
                    y=example["Speaking pace (illustrative)"],
                    name="Speaking pace · wpm",
                    mode="lines+markers",
                    line={"color": "#167c80"},
                ),
                secondary_y=False,
            )
            figure.add_trace(
                go.Scatter(
                    x=example["Take"],
                    y=example["Pauses (illustrative)"],
                    name="Pause count",
                    mode="lines+markers",
                    line={"color": "#cf8d48"},
                ),
                secondary_y=True,
            )
            figure.update_layout(title="An illustrative reflection view", height=330)
            figure.update_xaxes(title_text="Practice take")
            figure.update_yaxes(title_text="Words per minute", secondary_y=False)
            figure.update_yaxes(title_text="Pauses", secondary_y=True)
            chart(figure, "sample-progress")
            st.info("A useful practice goal is a clear idea and a comfortable breath—not a perfect chart.")
        return
    frame = pd.DataFrame(history)
    frame["Take"] = np.arange(1, len(frame) + 1)
    if frame["words_per_minute"].notna().any():
        figure = px.line(
            frame,
            x="Take",
            y="words_per_minute",
            markers=True,
            color_discrete_sequence=["#167c80"],
            title="Your speaking pace across saved takes",
        )
        figure.update_yaxes(title="Words per minute · transcript-based")
        chart(figure, "history-pace")
    figure = px.bar(
        frame,
        x="Take",
        y="pause_count",
        color_discrete_sequence=["#cf8d48"],
        title="Detected internal pauses",
    )
    chart(figure, "history-pauses")
    st.dataframe(
        frame.drop(columns=["Take"]),
        hide_index=True,
        width="stretch",
    )
    json_bytes = json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "Export practice notes (JSON)",
        json_bytes,
        "speakready-practice-notes.json",
        "application/json",
    )
    if st.button("Clear session history"):
        st.session_state.pop("speakready_history", None)
        st.rerun()
    st.caption("History contains summary metrics only—not audio or the transcript. It lasts for this app session.")


def render_about() -> None:
    st.markdown('<div class="sono-section-label">Use the numbers kindly</div>', unsafe_allow_html=True)
    st.header("What SpeakReady measures")
    st.write(
        "SpeakReady is for rehearsing everyday answers, talks and presentations. "
        "It offers small, observable prompts so practice can feel more actionable."
    )
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.subheader("What the feedback means")
            st.markdown(
                """
                - **Speaking pace** uses your own transcript and detected active time.
                - **Pauses** come from a simple frame-energy voice-activity heuristic.
                - **Pitch contour** tracks periodicity in voiced frames only.
                - **Loudness** uses the recording's digital level; microphones differ.
                - **Compression** is an optional soft-knee, feed-forward dynamics tool.
                """
            )
        with right:
            st.subheader("What it does not mean")
            st.markdown(
                """
                - It does not diagnose a speech, hearing or health condition.
                - It does not measure confidence, intelligence, accent quality or suitability.
                - It does not decide which pace, pitch or pauses are best for you.
                - Transcript-based pace depends on transcript accuracy.
                - The English-oriented pace examples are not universal language norms.
                """
            )
    st.subheader("Privacy, plainly")
    st.markdown(
        """
        **Local run:** audio analysis happens in your Python process; recordings are processed in memory and not added to history.
        **Hosted app:** audio is sent to Streamlit Community Cloud to run the app. Avoid uploading sensitive audio to the hosted service.
        **History:** stores only summary numbers in the current Streamlit session. Exported JSON is an explicit download.
        **Compressor:** creates a WAV download in memory; no server-side audio archive is created.
        """
    )
    st.subheader("The signal-processing idea")
    st.latex(r"\mathrm{WPM}=\frac{\text{transcript word count}}{\text{detected speaking seconds}}\times60")
    st.latex(r"g_{\mathrm{dB}}=L_{\mathrm{compressed}}-L_{\mathrm{input}}")
    st.caption("Activity thresholds and pitch tracking are heuristics. Room noise and input levels can affect them.")


st.sidebar.markdown("## SpeakReady")
st.sidebar.caption("Practice for what you want to say.")
st.sidebar.markdown(
    "**A gentle note:** this is a practice aid, not a clinical or identity tool."
)
st.title("SpeakReady")
st.caption("A private-by-default speaking-practice coach · record a take, reflect, try again")
practice_tab, progress_tab, compressor_tab, learn_tab = st.tabs(
    ["Practice", "My progress", "Audio compressor", "How it works"]
)
with practice_tab:
    render_practice()
with progress_tab:
    render_progress()
with compressor_tab:
    render_compressor()
with learn_tab:
    render_about()
