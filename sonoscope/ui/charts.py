"""Enhanced charts for SpeakReady: gauges, comparison views, annotated waveforms."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEAL = "#167c80"
AMBER = "#cf8d48"
INK = "#142e35"
MUTED = "#60777c"
GREEN = "#2e9e6b"
RED = "#c0504d"
BG = "rgba(0,0,0,0)"
GRID = "rgba(24,62,68,.08)"


def _base_layout(title: str, height: int = 300) -> go.Layout:
    return go.Layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        margin={"l": 25, "r": 15, "t": 55, "b": 35},
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"color": INK, "size": 12},
        showlegend=False,
    )


def wave_chart(signal: np.ndarray, sample_rate: int, title: str, key: str, speech_mask: np.ndarray | None = None) -> go.Figure:
    step = max(1, len(signal) // 12_000)
    x = np.arange(0, len(signal), step) / sample_rate
    y = signal[::step]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": TEAL, "width": 1.2},
            fill="tozeroy",
            fillcolor="rgba(22,124,128,.1)",
            name="audio",
        )
    )
    if speech_mask is not None:
        mask_step = max(1, len(speech_mask) // len(x))
        speech_x = x[::mask_step][: len(x)]
        speech_y = np.interp(np.arange(len(x)), np.arange(len(speech_mask))[::mask_step][: len(x)], speech_mask[::mask_step][: len(x)])
        figure.add_trace(
            go.Scatter(
                x=speech_x,
                y=speech_y,
                mode="lines",
                line={"color": AMBER, "width": 2},
                fill="tozeroy",
                fillcolor="rgba(207,141,72,.18)",
                name="detected speech",
            )
        )
    figure.update_layout(
        _base_layout(title),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        xaxis={"showgrid": True, "gridcolor": GRID},
        yaxis={"showgrid": True, "gridcolor": GRID},
    )
    return figure


def pace_gauge(wpm: float | None, target: tuple[int, int], title: str = "Speaking pace") -> go.Figure:
    low, high = target
    value = wpm if wpm is not None else 0
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 14}},
            delta={"reference": (low + high) / 2, "position": "top"},
            gauge={
                "shape": "angular",
                "axis": {"range": [None, max(260, high + 40)], "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": TEAL},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": GRID,
                "steps": [
                    {"range": [0, low], "color": "rgba(207,141,72,.12)"},
                    {"range": [low, high], "color": "rgba(22,124,128,.12)"},
                    {"range": [high, max(260, high + 40)], "color": "rgba(207,141,72,.12)"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 2},
                    "thickness": 0.75,
                    "value": (low + high) / 2,
                },
            },
        )
    )
    figure.update_layout(height=260, margin={"l": 20, "r": 20, "t": 60, "b": 20}, paper_bgcolor=BG, plot_bgcolor=BG)
    return figure


def pitch_range_chart(report: dict, target: tuple[int, int] | None = None) -> go.Figure:
    pitches = np.asarray([np.nan if v is None else v for v in report["pitch_values_hz"]], dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=report["pitch_times"],
            y=pitches,
            mode="lines+markers",
            connectgaps=False,
            marker={"size": 4, "color": AMBER},
            line={"color": AMBER, "width": 1.5},
            name="Estimated pitch",
        )
    )
    if report["median_pitch_hz"] is not None:
        figure.add_hline(
            y=report["median_pitch_hz"],
            line_dash="dash",
            line_color=TEAL,
            annotation_text=f"Median {report['median_pitch_hz']:.0f} Hz",
            annotation_position="top right",
        )
    if target:
        lo, hi = target
        figure.add_hrect(y0=lo * 0.55, y1=hi * 0.55, fillcolor=TEAL, opacity=0.08, line_width=0)
    figure.update_layout(
        _base_layout("Pitch movement · voiced frames", height=300),
        xaxis_title="Time (seconds)",
        yaxis_title="Estimated pitch (Hz)",
        xaxis={"showgrid": True, "gridcolor": GRID},
        yaxis={"showgrid": True, "gridcolor": GRID},
    )
    return figure


def pause_timeline_chart(report: dict) -> go.Figure:
    times = np.asarray(report["frame_times"])
    levels = np.asarray(report["frame_levels_dbfs"])
    speech = np.asarray(report["speech_mask"], dtype=bool)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=times,
            y=levels,
            mode="lines",
            line={"color": MUTED, "width": 1},
            name="Frame level",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times[speech],
            y=levels[speech],
            mode="markers",
            marker={"color": TEAL, "size": 4},
            name="Speech frames",
        )
    )
    figure.update_layout(
        _base_layout("Speaking rhythm · detected speech and pauses", height=340),
        xaxis_title="Time (seconds)",
        yaxis_title="RMS level (dBFS)",
        xaxis={"showgrid": True, "gridcolor": GRID},
        yaxis={"showgrid": True, "gridcolor": GRID},
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def comparison_chart(take_a: dict, take_b: dict, label_a: str = "Take A", label_b: str = "Take B") -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                           subplot_titles=(f"{label_a} — rhythm", f"{label_b} — rhythm"))
    for row, (take, label, color) in enumerate(((take_a, label_a, TEAL), (take_b, label_b, AMBER)), start=1):
        times = np.asarray(take["frame_times"])
        levels = np.asarray(take["frame_levels_dbfs"])
        speech = np.asarray(take["speech_mask"], dtype=bool)
        figure.add_trace(
            go.Scatter(x=times, y=levels, mode="lines", line={"color": MUTED, "width": 1}, showlegend=False),
            row=row, col=1,
        )
        figure.add_trace(
            go.Scatter(x=times[speech], y=levels[speech], mode="markers",
                       marker={"color": color, "size": 4}, name=label, showlegend=(row == 1)),
            row=row, col=1,
        )
    figure.update_layout(
        height=520,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"color": INK, "size": 12},
        margin={"l": 25, "r": 15, "t": 60, "b": 40},
        legend={"orientation": "h", "y": 1.02},
    )
    figure.update_xaxes(title_text="Time (seconds)", showgrid=True, gridcolor=GRID)
    figure.update_yaxes(title_text="RMS level (dBFS)", showgrid=True, gridcolor=GRID)
    return figure


def progress_lines(frame, title: str, y: str, color: str, ytitle: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["Take"],
            y=frame[y],
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "color": color},
            name=y,
        )
    )
    figure.update_layout(
        _base_layout(title, height=300),
        xaxis_title="Practice take",
        yaxis_title=ytitle,
        xaxis={"showgrid": True, "gridcolor": GRID},
        yaxis={"showgrid": True, "gridcolor": GRID},
    )
    return figure
