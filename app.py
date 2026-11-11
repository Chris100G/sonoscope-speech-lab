"""SONOSCOPE: a local-first, interactive speech-processing laboratory."""

from __future__ import annotations

import importlib.util
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sonoscope.assistant import (
    build_demo_recognizer,
    demo_commands,
    fuse_scores,
    recognize_with_fusion,
    synthesize_command,
)
from sonoscope.dsp.audio import load_audio
from sonoscope.dsp.distances import (
    cepstral_distance,
    cosine_distance,
    dtw,
    euclidean_distance,
    itakura_distance,
)
from sonoscope.dsp.features import (
    delta,
    frame_signal,
    mel_filterbank,
    mfcc,
    pre_emphasis,
    window_frames,
)
from sonoscope.dsp.hmm import DiscreteHMM, baum_welch
from sonoscope.dsp.speech import estimate_pitch, formants, lpc, lpc_resynthesis, vad
from sonoscope.recognizers import DTWTemplateRecognizer, VQHMMRecognizer
from sonoscope.ui.profile_io import export_profile, import_profile
from sonoscope.ui.style import apply_styles

SAMPLE_RATE = 16_000
COMMANDS = demo_commands()
COMMAND_BY_KEY = {command.key: command for command in COMMANDS}
_chart_counter = 0


st.set_page_config(
    page_title="SONOSCOPE | Speech Laboratory",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_styles()


@st.cache_resource
def demo_model() -> DTWTemplateRecognizer:
    """Cache the tiny, deterministic demo model for this app process."""
    return build_demo_recognizer(SAMPLE_RATE)


def audio_input(key: str, *, demo: bool = True) -> tuple[np.ndarray, int, str]:
    """Render upload/microphone controls and return audio standardized to 16 kHz."""
    uploaded = st.file_uploader(
        "Upload audio (WAV, FLAC, or supported MP3)",
        type=["wav", "flac", "mp3"],
        key=f"{key}-upload",
    )
    microphone = st.audio_input("Or record a short clip", key=f"{key}-mic")
    source = uploaded if uploaded is not None else microphone
    if source is not None:
        audio, rate = load_audio(source.getvalue(), target_sr=SAMPLE_RATE)
        if audio.size == 0:
            raise ValueError("The selected audio contains no samples.")
        return audio, rate, "recording" if microphone is source else "upload"
    if demo:
        selection = st.selectbox(
            "Synthetic demo pattern",
            COMMANDS,
            format_func=lambda command: f"{command.text} · {command.language.upper()}",
            key=f"{key}-demo",
        )
        return synthesize_command(selection), SAMPLE_RATE, "synthetic demo"
    return np.empty(0, dtype=np.float64), SAMPLE_RATE, "no audio"


def waveform_figure(signal: np.ndarray, sample_rate: int, title: str) -> go.Figure:
    """Build a bounded waveform plot for responsive rendering."""
    step = max(1, len(signal) // 12_000)
    times = np.arange(0, len(signal), step) / sample_rate
    figure = go.Figure(
        go.Scattergl(
            x=times,
            y=signal[::step],
            mode="lines",
            name="waveform",
            line={"color": "#167c80", "width": 1.4},
            fill="tozeroy",
            fillcolor="rgba(22,124,128,.10)",
        )
    )
    figure.update_layout(
        title={"text": title, "x": 0.025, "xanchor": "left"},
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin={"l": 25, "r": 15, "t": 55, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(24,62,68,.08)", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(24,62,68,.08)", zeroline=False)
    return figure


def feature_heatmap(features: np.ndarray, title: str) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(z=features.T, colorscale="Tealgrn", colorbar_title="Value")
    )
    figure.update_layout(
        title=title,
        xaxis_title="Frame",
        yaxis_title="Coefficient",
        height=320,
        margin={"l": 20, "r": 20, "t": 55, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def demo_spectrogram_figure(signal: np.ndarray, features: np.ndarray) -> go.Figure:
    """Create a compact time/frequency/cepstral overview of a demo signal."""
    frames = window_frames(frame_signal(signal, 400, 160))
    spectrum = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=10 * np.log10(np.maximum(spectrum[:, :97].T, 1e-12)),
            x=np.arange(len(frames)) * 0.01,
            y=frequencies[:97],
            colorscale="Viridis",
            colorbar={"title": "dB", "x": 1.0, "len": 0.42, "y": 0.77},
            xaxis="x",
            yaxis="y",
            name="Short-time spectrum",
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=features.T,
            x=np.arange(len(features)) * 0.01,
            colorscale="Teal",
            colorbar={"title": "MFCC", "x": 1.0, "len": 0.42, "y": 0.23},
            xaxis="x2",
            yaxis="y2",
            name="MFCC",
        )
    )
    figure.update_layout(
        title={"text": "A command, seen two ways", "x": 0.03, "xanchor": "left"},
        height=490,
        margin={"l": 22, "r": 68, "t": 58, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"domain": [0, 0.94], "title": "Time (s)", "anchor": "y"},
        yaxis={"domain": [0.54, 1], "title": "Frequency (Hz)", "anchor": "x"},
        xaxis2={"domain": [0, 0.94], "title": "Time (s)", "anchor": "y2"},
        yaxis2={"domain": [0, 0.46], "title": "Cepstral coefficient", "anchor": "x2"},
        showlegend=False,
    )
    return figure


def show_chart(figure: go.Figure, *, key: str | None = None) -> None:
    """Render a Plotly chart with a unique, stable element identity."""
    global _chart_counter
    _chart_counter += 1
    st.plotly_chart(figure, width="stretch", key=key or f"sonoscope-chart-{_chart_counter}")


def spectral_frames(signal: np.ndarray) -> np.ndarray:
    frames = window_frames(frame_signal(signal, 400, 160))
    if frames.size == 0:
        return np.empty((0, 201))
    return np.abs(np.fft.rfft(frames, n=400, axis=1)) ** 2


def signal_panels(signal: np.ndarray, source: str) -> np.ndarray:
    st.caption(f"Input: **{source}** · {len(signal) / SAMPLE_RATE:.2f} seconds · 16 kHz mono")
    show_chart(waveform_figure(signal, SAMPLE_RATE, "Waveform"))
    st.audio(signal, sample_rate=SAMPLE_RATE)
    return mfcc(signal, SAMPLE_RATE)


def render_home() -> None:
    st.markdown(
        """
        <section class="sono-hero">
          <div class="sono-kicker">THE INTERPRETABLE SPEECH LAB</div>
          <h1>SONOSCOPE</h1>
          <p>See speech as signal, features and decisions. Explore classical audio processing with every step visible.</p>
          <div class="sono-hero-meta">
            <span class="sono-chip">◉ CPU-ready</span>
            <span class="sono-chip">16 kHz audio</span>
            <span class="sono-chip">English · हिन्दी · ಕನ್ನಡ</span>
            <span class="sono-chip">No librosa</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Methods", "MFCC · DTW · HMM")
    c2.metric("Languages", "EN · HI · KN")
    c3.metric("Runtime", "Local CPU · 16 kHz")
    st.markdown(
        '<div class="sono-section-label">Explore the lab</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.85, 1.4], gap="large")
    with left:
        st.subheader("Listen & inspect")
        st.markdown(
            '<div class="sono-small-note">A generated acoustic pattern demonstrates the feature and recognition pipeline. '
            'It is intentionally synthetic—not spoken language.</div>',
            unsafe_allow_html=True,
        )
        language_names = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
        selected_language = st.selectbox(
            "Command language",
            ["English", "Hindi", "Kannada"],
            key="home-language",
        )
        language_code = {value: key for key, value in language_names.items()}[selected_language]
        language_commands = [command for command in COMMANDS if command.language == language_code]
        command = st.selectbox(
            "Demo phrase",
            language_commands,
            format_func=lambda item: item.text,
            key="home-command",
        )
        signal = synthesize_command(command)
        features = mfcc(signal, SAMPLE_RATE)
        model = demo_model()
        scores = model.score(features)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        recognition = model.recognize(features)
        c1, c2 = st.columns(2)
        c1.metric("DTW match", "Exact" if recognition.label == command.key else "Review")
        c2.metric("Frames", f"{len(features)}")
        if st.button("Run demo recognition", type="primary"):
            started = time.perf_counter()
            result = model.recognize(features)
            elapsed = (time.perf_counter() - started) * 1000
            if result.label == command.key:
                st.success(f"Matched **{command.text}** ({command.language.upper()})")
            else:
                st.warning(f"Model result: {result.label or 'rejected'}")
            st.caption(f"DTW score {result.score:.3f} · {elapsed:.1f} ms on this process")
        if ranked:
            relative_scores = pd.DataFrame(
                [
                    {
                        "Candidate": COMMAND_BY_KEY[key].text,
                        "Relative DTW score": float(np.exp(np.clip(score - ranked[0][1], -20, 0))),
                    }
                    for key, score in ranked[:5]
                ]
            )
            ranking_chart = px.bar(
                relative_scores.sort_values("Relative DTW score"),
                x="Relative DTW score",
                y="Candidate",
                orientation="h",
                color="Relative DTW score",
                color_continuous_scale=["#d9ece7", "#167c80"],
                title="Nearest command templates",
            )
            ranking_chart.update_layout(
                height=255,
                margin={"l": 5, "r": 10, "t": 50, "b": 15},
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            show_chart(ranking_chart)
    with right:
        show_chart(demo_spectrogram_figure(signal, features))
    with st.container(border=True):
        lab1, lab2 = st.columns([1.1, 1])
        with lab1:
            st.markdown('<div class="sono-section-label">One signal, three views</div>', unsafe_allow_html=True)
            st.subheader("From waveform to model")
            st.markdown(
                "The waveform is split into short, overlapping frames. A mel filterbank summarizes "
                "spectral energy, and the discrete cosine transform gives a compact MFCC sequence."
            )
            st.markdown(
                '<div class="sono-equation-card">Pre-emphasis &nbsp; <code>y[n] = x[n] − α x[n−1]</code></div>'
                '<div class="sono-equation-card">MFCC &nbsp; <code>log mel energy → DCT → cepstral features</code></div>'
                '<div class="sono-equation-card">DTW &nbsp; <code>minimum-cost monotonic frame alignment</code></div>',
                unsafe_allow_html=True,
            )
        with lab2:
            st.markdown('<div class="sono-section-label">Choose an experiment</div>', unsafe_allow_html=True)
            st.markdown(
                """
                - **01 · Signal X-Ray** — waveform, spectrum and cepstral dynamics
                - **02 · Vocal Tract** — LPC, pitch, formants and resynthesis
                - **03 · Distance Lab** — DTW path and feature distances
                - **04 · HMM Lab** — forward, backward and Viterbi
                - **05 · Voice Assistant** — enroll, compare, confirm
                """
            )
    st.warning(
        "Research/demo tool, not an identity or safety system. Similarity scores are not calibrated probabilities; "
        "always confirm actions and use consenting recordings."
    )


def render_signal_xray() -> None:
    st.header("Signal X-Ray")
    st.write("Inspect the analysis chain from waveform to cepstral dynamics.")
    signal, _, source = audio_input("xray")
    features = signal_panels(signal, source)
    show_chart(feature_heatmap(features, "MFCC"))
    first_delta = delta(features)
    second_delta = delta(first_delta)
    a, b = st.columns(2)
    with a:
        show_chart(feature_heatmap(first_delta, "Delta MFCC"))
    with b:
        show_chart(feature_heatmap(second_delta, "Delta-delta MFCC"))
    emphasized = pre_emphasis(signal)
    frames = window_frames(frame_signal(emphasized, 400, 160))
    if frames.size:
        spectrum = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
        bank = mel_filterbank(SAMPLE_RATE, 512, n_mels=26)
        mel_energy = spectrum @ bank.T
        col1, col2 = st.columns(2)
        with col1:
            show_chart(waveform_figure(emphasized, SAMPLE_RATE, "Pre-emphasized signal"))
        with col2:
            show_chart(feature_heatmap(np.log(np.maximum(mel_energy, 1e-12)), "Log mel-filterbank energy"))
    with st.expander("What the transform does"):
        st.markdown(
            "Pre-emphasis boosts spectral tilt; overlapping Hamming frames approximate local stationarity. "
            "The power spectrum is integrated by triangular mel filters, log-compressed, then decorrelated "
            "with a DCT. Delta and delta-delta coefficients summarize local temporal change."
        )
        st.latex(r"c_m=\sum_{k=1}^{K}\log(E_k)\cos\left[\frac{\pi m}{K}(k-\frac12)\right]")


def render_vocal_tract() -> None:
    st.header("Vocal Tract Studio")
    st.write("Estimate a source-filter model and inspect resonance and periodicity.")
    signal, _, source = audio_input("vocal")
    features = signal_panels(signal, source)
    if len(signal) <= 16:
        st.warning("Provide a longer clip to estimate LPC coefficients.")
        return
    order = st.slider("LPC order", min_value=8, max_value=24, value=16, step=2)
    coefficients = lpc(signal[: min(len(signal), 4_000)], order)
    pitch = estimate_pitch(signal, SAMPLE_RATE)
    resonance = formants(signal[: min(len(signal), 4_000)], SAMPLE_RATE, order)
    active = vad(signal)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pitch estimate", f"{pitch:.1f} Hz" if pitch else "Unvoiced / unclear")
    c2.metric("LPC order", str(order))
    c3.metric("Active frames", f"{int(active.sum())} / {len(active)}")
    st.write("Candidate formants (Hz):", ", ".join(f"{value:.0f}" for value in resonance[:5]) or "none detected")
    a, b = st.columns(2)
    with a:
        show_chart(feature_heatmap(features, "MFCC context"))
    with b:
        if len(resonance) >= 2:
            user_point = go.Scatter(x=[resonance[1]], y=[resonance[0]], mode="markers+text", text=["input"], name="Input")
        else:
            user_point = go.Scatter(x=[], y=[], mode="markers", name="Input")
        reference = pd.DataFrame(
            {"Vowel": ["i", "e", "a", "o", "u"], "F2 (Hz)": [2500, 2100, 1500, 1000, 800], "F1 (Hz)": [300, 500, 900, 600, 350]}
        )
        chart = px.scatter(reference, x="F2 (Hz)", y="F1 (Hz)", text="Vowel", title="Illustrative vowel space")
        chart.add_trace(user_point)
        chart.update_xaxes(autorange="reversed")
        chart.update_yaxes(autorange="reversed")
        show_chart(chart)
    impulse = np.zeros(2_000)
    impulse[:: max(1, int(SAMPLE_RATE / (pitch or 120.0)))] = 0.15
    resynthesized = lpc_resynthesis(impulse, coefficients)
    show_chart(waveform_figure(resynthesized, SAMPLE_RATE, "LPC source-filter resynthesis"))
    st.caption("Formant points are rough resonances, not a validated phonetic classifier. Vowel references are illustrative.")


def render_distance_lab() -> None:
    st.header("Distance Lab")
    st.write("Compare feature sequences and inspect the monotonic DTW alignment path.")
    left, right = st.columns(2)
    with left:
        first = st.selectbox("Reference demo", COMMANDS, index=0, format_func=lambda item: f"{item.text} ({item.language})", key="dist-ref")
    with right:
        second = st.selectbox("Comparison demo", COMMANDS, index=1, format_func=lambda item: f"{item.text} ({item.language})", key="dist-cmp")
    x = mfcc(synthesize_command(first), SAMPLE_RATE)
    y = mfcc(synthesize_command(second), SAMPLE_RATE)
    metric = st.selectbox("DTW local cost", ["euclidean", "cosine", "cepstral", "itakura"])
    if metric == "itakura":
        x_cost = spectral_frames(synthesize_command(first))
        y_cost = spectral_frames(synthesize_command(second))
    else:
        x_cost, y_cost = x, y
    cost, path = dtw(x_cost, y_cost, metric)
    static = {
        "Euclidean (mean MFCC)": euclidean_distance(x.mean(axis=0), y.mean(axis=0)),
        "Cosine (mean MFCC)": cosine_distance(x.mean(axis=0), y.mean(axis=0)),
        "Cepstral": cepstral_distance(x.mean(axis=0), y.mean(axis=0)),
        "Itakura-Saito (mean power spectrum)": itakura_distance(spectral_frames(synthesize_command(first)).mean(axis=0), spectral_frames(synthesize_command(second)).mean(axis=0)),
    }
    a, b, c = st.columns(3)
    a.metric("DTW normalized cost", f"{cost:.3f}")
    b.metric("Alignment path length", str(len(path)))
    selected_static_key = {
        "euclidean": "Euclidean (mean MFCC)",
        "cosine": "Cosine (mean MFCC)",
        "cepstral": "Cepstral",
        "itakura": "Itakura-Saito (mean power spectrum)",
    }[metric]
    c.metric("Selected static distance", f"{static[selected_static_key]:.3f}")
    path_array = np.asarray(path)
    figure = go.Figure(
        go.Scatter(x=path_array[:, 1], y=path_array[:, 0], mode="lines", line={"color": "#1e6d74"})
    )
    figure.update_layout(title="DTW alignment path", xaxis_title="Comparison frame", yaxis_title="Reference frame", height=380)
    show_chart(figure)
    st.dataframe(pd.DataFrame({"Distance": list(static), "Value": list(static.values())}), hide_index=True, width="stretch")
    st.caption("Lower is closer. Itakura-Saito is applied to non-negative power spectra; the other static comparisons use MFCCs.")


def render_hmm_lab() -> None:
    st.header("HMM Lab")
    st.write("Explore the forward/backward recurrences, Viterbi decoding, and Baum-Welch updates.")
    initial = np.array([0.6, 0.4])
    transition = np.array([[0.7, 0.3], [0.2, 0.8]])
    emission = np.array([[0.9, 0.1, 0.0], [0.05, 0.45, 0.5]])
    observations = np.array([0, 1, 1, 2, 2, 1])
    model = DiscreteHMM(initial, transition, emission)
    forward_score, alpha = model.forward(observations)
    backward_score, beta = model.backward(observations)
    path_score, path = model.viterbi(observations)
    a, b, c = st.columns(3)
    a.metric("Forward log-likelihood", f"{forward_score:.3f}")
    b.metric("Backward log-likelihood", f"{backward_score:.3f}")
    c.metric("Viterbi log-probability", f"{path_score:.3f}")
    col1, col2 = st.columns(2)
    with col1:
        show_chart(feature_heatmap(alpha, "Forward state posterior (scaled)"))
        show_chart(feature_heatmap(beta, "Backward values (scaled)"))
    with col2:
        show_chart(feature_heatmap(transition, "Transition matrix A"))
        show_chart(feature_heatmap(emission, "Emission matrix B"))
    st.write("Observation symbols:", observations.tolist())
    st.write("Most likely state path:", path.tolist())
    if importlib.util.find_spec("hmmlearn") is None:
        st.caption("Optional hmmlearn baseline is not installed; the displayed inference is the from-scratch implementation.")
    else:
        st.caption("hmmlearn is installed but unused here so the displayed inference remains fully inspectable.")
    if st.button("Run Baum-Welch on a small demo corpus"):
        trained = baum_welch(
            [observations, np.array([0, 0, 1, 2, 2])],
            n_states=2,
            n_symbols=3,
            iterations=15,
            seed=7,
        )
        st.success(f"Trained model log-likelihood: {trained.score(observations):.3f}")
        show_chart(feature_heatmap(trained.transition, "Learned transition matrix"))
    with st.expander("Interpretation"):
        st.markdown(
            "Forward sums probabilities over all state paths; backward sums future observations. "
            "Viterbi uses max-product decoding. Baum-Welch alternates expected state/transition counts "
            "with normalized parameter updates. The implementation uses scaling/log likelihoods to control underflow."
        )
        st.latex(r"\alpha_t(j)=b_j(o_t)\sum_i \alpha_{t-1}(i)a_{ij}")


def _session_profile() -> dict[str, list[np.ndarray]]:
    profile = st.session_state.setdefault("enrollment", {})
    return profile


def _profile_json_shape(profile: dict[str, list[np.ndarray]]) -> dict[str, list[list[list[float]]]]:
    return {
        label: [sequence.tolist() for sequence in sequences]
        for label, sequences in profile.items()
    }


def render_voice_assistant() -> None:
    st.header("Voice Assistant")
    st.write("Enroll short voice templates, compare with DTW and VQ-HMM, then confirm any proposed action.")
    profile = _session_profile()
    audio, _, source = audio_input("assistant")
    features = mfcc(audio, SAMPLE_RATE)
    if not len(features):
        st.warning("No speech frames were produced for this input.")
        return
    st.caption(f"Input: {source} · {len(features)} MFCC frames")
    enroll_tab, recognize_tab, todo_tab = st.tabs(["Enrollment & learning", "Recognition", "To-do actions"])
    with enroll_tab:
        language = st.selectbox("Enrollment language", ["English", "Hindi", "Kannada"])
        label = st.text_input("Command phrase / label", placeholder="For example: add groceries")
        if st.button("Enroll this voice sample", type="primary"):
            if not label.strip():
                st.error("Enter a phrase label before enrolling.")
            else:
                profile.setdefault(label.strip(), []).append(features.copy())
                st.success(f"Added one sample to **{label.strip()}** ({language}).")
        st.write(f"Enrolled labels: {len(profile)} · examples: {sum(map(len, profile.values()))}")
        if profile:
            st.dataframe(
                pd.DataFrame([{"Command": name, "Samples": len(samples)} for name, samples in profile.items()]),
                hide_index=True,
                width="stretch",
            )
        export_data = export_profile(_profile_json_shape(profile))
        st.download_button("Export voice profile ZIP", export_data, "sonoscope-profile.zip", "application/zip")
        imported = st.file_uploader("Import profile ZIP", type=["zip"], key="profile-import")
        if imported is not None and st.button("Validate and import profile"):
            try:
                restored = import_profile(imported.getvalue())
            except ValueError as error:
                st.error(str(error))
            else:
                for name, sequences in restored.items():
                    profile.setdefault(name, []).extend(np.asarray(sequence, dtype=float) for sequence in sequences)
                st.success(f"Imported {len(restored)} command labels into this session.")
    with recognize_tab:
        source_commands = profile if profile else {
            command.key: [mfcc(synthesize_command(command), SAMPLE_RATE)] for command in COMMANDS
        }
        source_name = "your enrollment" if profile else "built-in synthetic examples"
        st.caption(f"Models are trained from {source_name}. At least two distinct labels are needed for meaningful fusion.")
        minimum_margin = st.slider("Minimum top-score margin for acceptance", 0.0, 2.0, 0.15, 0.05)
        if st.button("Recognize with DTW + VQ-HMM", type="primary"):
            if len(source_commands) < 2:
                st.warning("Enroll at least two distinct command labels before recognition.")
            else:
                with st.spinner("Extracting features and scoring local models..."):
                    dtw_model = DTWTemplateRecognizer().fit(source_commands)
                    hmm_model = VQHMMRecognizer(
                        codebook_size=8,
                        n_states=3,
                        iterations=8,
                        seed=17,
                    ).fit(source_commands)
                    score_sets = {
                        "dtw": dtw_model.score(features),
                        "vq_hmm": hmm_model.score(features),
                    }
                    fused = fuse_scores(score_sets)
                    result = recognize_with_fusion(score_sets, threshold=0.0)
                ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
                margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
                accepted = result.label is not None and not result.rejected and margin >= minimum_margin
                st.session_state["last_recognition"] = {
                    "label": result.label if accepted else None,
                    "candidate": result.label,
                    "features": features.copy(),
                    "scores": fused,
                    "margin": margin,
                    "accepted": accepted,
                }
        last_result = st.session_state.get("last_recognition")
        if last_result:
            if not last_result["accepted"] or last_result["label"] is None:
                st.warning("No command passed the confidence gate. Record again or add enrollment examples.")
            else:
                candidate = last_result["label"]
                display_label = COMMAND_BY_KEY[candidate].text if candidate in COMMAND_BY_KEY else candidate
                st.success(f"Candidate: **{display_label}** · normalized score margin {last_result['margin']:.2f}")
                st.caption("Scores are relative z-normalized values, not calibrated probabilities.")
                show_chart(px.bar(x=list(last_result["scores"]), y=list(last_result["scores"].values()), labels={"x": "Label", "y": "Fused score"}, title="Relative fused scores"))
            if last_result["candidate"] is not None:
                available = sorted(set(source_commands) | {last_result["candidate"]})
                correction = st.selectbox("Correction-based learning: choose the correct label", available, key="correction")
                if st.button("Save correction as a new enrollment example"):
                    profile.setdefault(correction, []).append(last_result["features"])
                    st.success(f"Added corrected example to **{correction}**.")
        whisper_available = importlib.util.find_spec("whisper") is not None
        if st.checkbox("Optional local Whisper tiny baseline"):
            if not whisper_available:
                st.info("Whisper is not installed. Core recognition remains fully local and available.")
            else:
                st.warning("Whisper may download its model on first use; continue only if that transfer is acceptable.")
                if st.button("Transcribe locally with Whisper tiny"):
                    import whisper

                    model = whisper.load_model("tiny", device="cpu")
                    transcription = model.transcribe(audio.astype(np.float32), fp16=False)
                    st.write(transcription["text"])
        if st.checkbox("Optional local text-to-speech preview (pyttsx3)"):
            if importlib.util.find_spec("pyttsx3") is None:
                st.info("pyttsx3 is not installed. Voice recognition and enrollment do not require it.")
            else:
                speech_text = st.text_input("Text to speak", value=label or "SONOSCOPE voice preview")
                if st.button("Generate local TTS preview"):
                    import pyttsx3

                    with tempfile.TemporaryDirectory(prefix="sonoscope-") as directory:
                        output_path = Path(directory) / "preview.wav"
                        engine = pyttsx3.init()
                        engine.save_to_file(speech_text, str(output_path))
                        engine.runAndWait()
                        st.audio(output_path.read_bytes(), format="audio/wav")
    with todo_tab:
        st.caption("Recognized commands only propose changes. Confirm explicitly before modifying this session's list.")
        todos = st.session_state.setdefault("todos", [])
        task_text = st.text_input("Task to add", placeholder="e.g., send the project draft")
        last_result = st.session_state.get("last_recognition", {})
        recognized = last_result.get("label")
        intent = COMMAND_BY_KEY[recognized].intent if recognized in COMMAND_BY_KEY else ""
        if recognized and recognized.lower().startswith(("add ", "add:", "todo_add")):
            intent = "todo_add"
        elif recognized and recognized.lower().startswith(("complete ", "done ", "todo_done")):
            intent = "todo_done"
        suggested = {"todo_add": "Add task", "todo_done": "Complete a task"}.get(intent)
        if suggested:
            st.info(f"Recognized command proposes: **{suggested}**. Confirm before applying.")
        selected = st.selectbox("Action", ["Add task", "Complete a task"], index=0 if suggested != "Complete a task" else 1)
        if selected == "Complete a task" and todos:
            target = st.selectbox("Task", list(range(len(todos))), format_func=lambda i: todos[i]["text"])
        else:
            target = None
        confirm = st.checkbox("I confirm this to-do change")
        if st.button("Apply confirmed action", disabled=not confirm):
            if selected == "Add task":
                if not task_text.strip():
                    st.error("Enter a task before adding it.")
                else:
                    todos.append({"text": task_text.strip(), "done": False})
                    st.success("Task added to this session.")
            elif target is None:
                st.warning("There are no tasks to complete.")
            else:
                todos[target]["done"] = True
                st.success("Task marked complete.")
        if todos:
            st.dataframe(pd.DataFrame(todos), hide_index=True, width="stretch")
        else:
            st.info("No tasks yet.")
    st.caption("Local mode does not persist audio. Exported profiles contain MFCC voice features; treat them as sensitive.")


def render_benchmark_ethics() -> None:
    st.header("Benchmark & Ethics")
    st.write("Run an explicitly synthetic sanity-check benchmark and inspect the privacy boundary.")
    if st.button("Evaluate synthetic command set"):
        matrix = np.zeros((len(COMMANDS), len(COMMANDS) + 1), dtype=int)
        labels = [command.key for command in COMMANDS]
        elapsed: list[float] = []
        model = demo_model()
        for index, command in enumerate(COMMANDS):
            audio = synthesize_command(command)
            started = time.perf_counter()
            result = model.recognize(mfcc(audio, SAMPLE_RATE))
            elapsed.append((time.perf_counter() - started) * 1000)
            predicted_index = labels.index(result.label) if result.label in labels else len(COMMANDS)
            matrix[index, predicted_index] += 1
        accuracy = float(np.trace(matrix[:, :len(COMMANDS)]) / matrix.sum())
        first, second = st.columns(2)
        first.metric("Synthetic command accuracy", f"{accuracy:.0%}")
        second.metric("Median latency", f"{np.median(elapsed):.1f} ms")
        figure = px.imshow(matrix, x=labels + ["rejected"], y=labels, text_auto=True, color_continuous_scale="Tealgrn", title="Synthetic confusion matrix")
        figure.update_layout(xaxis_title="Predicted label", yaxis_title="Reference label")
        show_chart(figure)
        rng = np.random.default_rng(2025)
        noise_rows = []
        for snr_db in (20, 10, 0):
            correct = 0
            for command in COMMANDS:
                clean = synthesize_command(command)
                power = np.sqrt(np.mean(clean**2))
                noise = rng.normal(0.0, power / (10 ** (snr_db / 20)), clean.shape)
                result = model.recognize(mfcc(clean + noise, SAMPLE_RATE))
                correct += result.label == command.key
            noise_rows.append({"Condition": f"{snr_db} dB SNR", "Accuracy": correct / len(COMMANDS)})
        speaker_correct = 0
        for command in COMMANDS:
            clean = synthesize_command(command)
            shifted = np.interp(
                np.arange(0, len(clean), 1.04),
                np.arange(len(clean)),
                clean,
            )
            result = model.recognize(mfcc(shifted, SAMPLE_RATE))
            speaker_correct += result.label == command.key
        noise_rows.append({"Condition": "synthetic pitch/time shift", "Accuracy": speaker_correct / len(COMMANDS)})
        show_chart(px.bar(pd.DataFrame(noise_rows), x="Condition", y="Accuracy", range_y=[0, 1], title="Synthetic noise / speaker-shift proxy"))
        st.caption("Generated tone-pattern sanity checks only. Pitch/time shift is not a real cross-speaker evaluation.")
    st.subheader("Robustness protocol")
    st.markdown(
        "- Measure accuracy and rejection by command, language, speaker, and recording condition.\n"
        "- Evaluate noisy audio at stated SNRs and report latency on the target CPU.\n"
        "- Keep speakers disjoint between enrollment and test sets; synthetic data cannot establish cross-speaker performance.\n"
        "- Do not treat z-normalized recognition scores as probabilities."
    )
    st.subheader("Privacy and ethics")
    st.markdown(
        "Local processing is the default. Recordings are held only in the current request/session and are not written to disk. "
        "A profile ZIP is a deliberate user export and includes voice-derived features. The optional Whisper baseline is a local "
        "model when installed; first-time model retrieval may access the network. Voice similarity is not authentication."
    )


def render_pronunciation_coach() -> None:
    st.header("Pronunciation Coach")
    st.write("Compare a reference recording and an attempt using MFCC + DTW. This is a timing/similarity aid, not phonetic grading.")
    a, b = st.columns(2)
    with a:
        reference_file = st.file_uploader("Reference WAV / FLAC / MP3", type=["wav", "flac", "mp3"], key="coach-ref")
    with b:
        attempt_file = st.file_uploader("Attempt WAV / FLAC / MP3", type=["wav", "flac", "mp3"], key="coach-attempt")
    if reference_file is None or attempt_file is None:
        st.info("Provide two clips to compare. Synthetic examples are intentionally not treated as spoken pronunciation.")
        return
    reference, _ = load_audio(reference_file.getvalue(), target_sr=SAMPLE_RATE)
    attempt, _ = load_audio(attempt_file.getvalue(), target_sr=SAMPLE_RATE)
    ref_features, attempt_features = mfcc(reference, SAMPLE_RATE), mfcc(attempt, SAMPLE_RATE)
    score, path = dtw(ref_features, attempt_features, "euclidean")
    x, y = st.columns(2)
    with x:
        show_chart(waveform_figure(reference, SAMPLE_RATE, "Reference waveform"))
    with y:
        show_chart(waveform_figure(attempt, SAMPLE_RATE, "Attempt waveform"))
    st.metric("Length-normalized DTW feature cost", f"{score:.3f}")
    path_data = np.asarray(path)
    figure = go.Figure(go.Scatter(x=path_data[:, 1], y=path_data[:, 0], mode="lines"))
    figure.update_layout(title="Frame alignment", xaxis_title="Attempt frame", yaxis_title="Reference frame", height=350)
    show_chart(figure)
    st.caption("A lower cost means closer MFCC sequences under this configuration; it does not identify mispronounced phonemes.")


st.sidebar.markdown("## SONOSCOPE")
st.sidebar.caption("Speech laboratory · 16 kHz · CPU")
st.sidebar.markdown("**Privacy:** local processing; no audio uploads to a server in local mode.")
tab_names = [
    "Home",
    "Signal X-Ray",
    "Vocal Tract Studio",
    "Distance Lab",
    "HMM Lab",
    "Voice Assistant",
    "Benchmark & Ethics",
    "Pronunciation Coach",
]
tabs = st.tabs(tab_names)
renderers = [
    render_home,
    render_signal_xray,
    render_vocal_tract,
    render_distance_lab,
    render_hmm_lab,
    render_voice_assistant,
    render_benchmark_ethics,
    render_pronunciation_coach,
]
for tab, renderer in zip(tabs, renderers, strict=True):
    with tab:
        renderer()
