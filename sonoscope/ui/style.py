"""Shared visual styling and small presentation helpers."""

import streamlit as st


def apply_styles() -> None:
    """Apply the SONOSCOPE visual system to the current Streamlit page."""
    st.markdown(
        """
        <style>
        :root {
            --sono-ink: #142e35;
            --sono-muted: #60777c;
            --sono-teal: #167c80;
            --sono-line: #e0ebe8;
            --sono-canvas: #f5f8f6;
        }
        .stApp { background: var(--sono-canvas); }
        .block-container { max-width: 1480px; padding: 2rem 2.4rem 4rem; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102f37 0%, #163e44 100%);
        }
        [data-testid="stSidebar"] * { color: #e9f4f1; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #b6d2ce; }
        [data-testid="stSidebar"] hr { border-color: #31555a; }
        h1, h2, h3 { color: var(--sono-ink); letter-spacing: -.025em; }
        h2 { margin-top: .5rem; }
        p, li { color: #36545a; }
        [data-testid="stMetric"] {
            min-height: 104px;
            background: linear-gradient(145deg, #ffffff 0%, #f6fbf9 100%);
            border: 1px solid var(--sono-line);
            border-radius: 16px;
            padding: 17px 19px;
            box-shadow: 0 5px 18px rgba(21, 61, 66, .045);
        }
        [data-testid="stMetricLabel"] p {
            color: var(--sono-muted); font-weight: 650; letter-spacing: .025em;
        }
        [data-testid="stMetricValue"] { color: var(--sono-ink); }
        [data-testid="stTabs"] [role="tablist"] {
            gap: .35rem; border-bottom: 1px solid var(--sono-line);
        }
        [data-testid="stTabs"] button {
            color: #47656a; font-weight: 650;
            border-radius: 10px 10px 0 0; padding: .7rem .95rem;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--sono-teal); border-bottom: 3px solid var(--sono-teal);
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff; border-color: var(--sono-line) !important;
            border-radius: 16px; box-shadow: 0 4px 16px rgba(21, 61, 66, .035);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(100deg, #176f73, #218d84);
            border: 0; border-radius: 10px; font-weight: 700;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(100deg, #125b61, #16766f); border: 0;
        }
        .sono-hero {
            position: relative; overflow: hidden;
            background:
              radial-gradient(ellipse at 86% 20%, rgba(128, 222, 194, .28), transparent 34%),
              linear-gradient(118deg, #102f37 0%, #14555b 57%, #167c80 100%);
            color: #fff; padding: 2.4rem 2.7rem; border-radius: 24px;
            margin: .25rem 0 1.1rem;
            box-shadow: 0 18px 38px rgba(18, 68, 72, .14);
            animation: sono-rise .45s ease-out both;
        }
        .sono-hero:after {
            content: ""; position: absolute; width: 240px; height: 240px;
            right: -54px; bottom: -156px; border: 1px solid rgba(255,255,255,.22);
            border-radius: 50%; box-shadow: 0 0 0 22px rgba(255,255,255,.045),
              0 0 0 48px rgba(255,255,255,.035), 0 0 0 76px rgba(255,255,255,.025);
        }
        .sono-hero h1 {
            color: #fff; margin: .25rem 0 0;
            font-size: clamp(2.4rem, 5vw, 3.5rem); letter-spacing: -.045em;
        }
        .sono-hero p {
            color: #d9efea; margin: .65rem 0 0;
            max-width: 720px; font-size: 1.08rem; line-height: 1.65;
        }
        .sono-kicker {
            color: #91e0c3; font-size: .73rem;
            font-weight: 750; letter-spacing: .15em;
        }
        .sono-hero-meta { margin-top: 1.4rem; display: flex; flex-wrap: wrap; gap: .55rem; }
        .sono-chip {
            color: #eaf8f3; background: rgba(235,255,248,.1);
            border: 1px solid rgba(235,255,248,.19); border-radius: 999px;
            padding: .35rem .7rem; font-size: .79rem; font-weight: 650;
        }
        .sono-section-label {
            color: var(--sono-teal); font-size: .72rem;
            font-weight: 750; letter-spacing: .12em; text-transform: uppercase;
        }
        .sono-small-note { color: var(--sono-muted); font-size: .86rem; line-height: 1.55; }
        .sono-equation-card {
            background: #edf5f2; border-left: 3px solid #218d84;
            border-radius: 0 10px 10px 0; padding: .45rem .8rem; margin: .5rem 0;
        }
        @keyframes sono-rise {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 720px) {
            .block-container { padding: 1.1rem 1rem 2.2rem; }
            .sono-hero { padding: 1.55rem 1.4rem; border-radius: 18px; }
            [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
