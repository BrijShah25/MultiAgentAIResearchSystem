"""
Research Desk - Streamlit interface for the multi-agent research pipeline
==========================================================================

Runs the same four stages as `pipeline.py`, one at a time, so progress can be
shown live:

    1. Search agent  - finds recent, reliable sources
    2. Reader agent  - scrapes the most relevant page
    3. Writer chain  - drafts the report
    4. Critic chain  - reviews the draft

Run it with:

    streamlit run app.py

Requires Streamlit 1.40 or newer.
"""

from __future__ import annotations

import html
import inspect
import json
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

# ---------------------------------------------------------------------------
# Agent imports
# ---------------------------------------------------------------------------
# Imported defensively so a missing API key or package shows a readable
# message in the UI instead of a raw traceback.
try:
    from agents import (
        build_reader_agent,
        build_search_agent,
        critic_chain,
        writer_chain,
    )

    AGENTS_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # noqa: BLE001
    AGENTS_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Desk",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Streamlit version compatibility
# ---------------------------------------------------------------------------
# Newer Streamlit replaces `use_container_width=True` with `width="stretch"`.
# Pick whichever this install understands so the app works on both.
def _stretch_kwargs() -> Dict[str, Any]:
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


STRETCH: Dict[str, Any] = _stretch_kwargs()


def h(markup: str) -> str:
    """Strip indentation so Markdown never mistakes nested HTML for a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines())


# ---------------------------------------------------------------------------
# Icons (inline SVG, 24x24, drawn with currentColor)
# ---------------------------------------------------------------------------
ICON_PATHS: Dict[str, str] = {
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l5 5"/>',
    "reader": '<path d="M6 3h8l5 5v13H6z"/><path d="M14 3v5h5"/><path d="M9 13h7M9 17h7"/>',
    "writer": '<path d="M4 20l1-4L16.5 4.5a2 2 0 0 1 3 3L8 19z"/><path d="M14 7l3 3"/>',
    "critic": '<path d="M12 3l8 3v6c0 4.5-3.2 7.7-8 9-4.8-1.3-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5L16 9.5"/>',
}


def icon(name: str, size: int = 22, stroke: float = 1.7) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{ICON_PATHS[name]}</svg>'
    )


LOGO_SVG = h(
    """
    <svg width="38" height="38" viewBox="0 0 40 40" aria-hidden="true">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#2CB5BE"/><stop offset="1" stop-color="#0E5A9C"/>
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="11" fill="url(#lg)"/>
      <path d="M11 27L17 14L23 24L29 12" fill="none" stroke="#fff" stroke-width="2.2"
            stroke-linecap="round" stroke-linejoin="round" opacity=".85"/>
      <circle cx="11" cy="27" r="2.8" fill="#fff"/><circle cx="17" cy="14" r="2.8" fill="#fff"/>
      <circle cx="23" cy="24" r="2.8" fill="#fff"/><circle cx="29" cy="12" r="2.8" fill="#fff"/>
    </svg>
    """
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

:root {
    --paper: #F3F6FA;
    --surface: #FFFFFF;
    --ink: #0F1B33;
    --ink-soft: #4A5875;
    --ink-faint: #8591A8;
    --line: #E1E6EF;
    --line-strong: #CBD3E1;
    --teal: #0E7C86;
    --teal-bright: #2CB5BE;
    --teal-soft: #E3F3F4;
    --indigo: #4F5FD5;
    --amber: #C77D0A;
    --amber-soft: #FDF2DD;
    --rose: #B4344A;
    --rose-soft: #FBE9EC;
    --green: #2B8A57;
    --green-soft: #E4F4EB;
    --shadow-sm: 0 1px 2px rgba(15,27,51,0.05), 0 1px 1px rgba(15,27,51,0.03);
    --shadow-md: 0 8px 24px -8px rgba(15,27,51,0.14), 0 2px 6px rgba(15,27,51,0.05);
}

html, body, .stApp, [class*="css"] {
    font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--paper); color: var(--ink); }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1180px; }

/* Visible keyboard focus everywhere */
button:focus-visible, a:focus-visible, input:focus-visible {
    outline: 3px solid rgba(14,124,134,0.45) !important;
    outline-offset: 2px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: #0E1830; border-right: 1px solid #1B2850; }
section[data-testid="stSidebar"] * { color: #D3DBEE; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF; font-weight: 700; }
section[data-testid="stSidebar"] .stButton > button {
    background: #16224A; color: #E8EDF9; border: 1px solid #28386A;
    text-align: left; justify-content: flex-start;
    font-weight: 500; font-size: 0.85rem; border-radius: 10px; min-height: 2.4rem;
}
section[data-testid="stSidebar"] .stButton > button:hover { border-color: var(--teal-bright); color: #FFFFFF; }
.side-brand { display: flex; align-items: center; gap: 0.7rem; margin: 0.3rem 0 0.9rem 0; }
.side-brand .t { font-size: 1.12rem; font-weight: 800; color: #FFFFFF; line-height: 1.1; }
.side-brand .s { font-size: 0.76rem; color: #8F9CBD; margin-top: 2px; }
.side-label { font-size: 0.8rem; font-weight: 700; color: #8F9CBD; margin: 1.3rem 0 0.5rem 0; }
.side-note { font-size: 0.82rem; line-height: 1.55; color: #A5B1D0; }
.side-status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.84rem; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #46C486; box-shadow: 0 0 0 3px rgba(70,196,134,0.2); }
.dot.bad { background: #E5677D; box-shadow: 0 0 0 3px rgba(229,103,125,0.2); }

/* ---------- Top bar ---------- */
.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.1rem; }
.brand { display: flex; align-items: center; gap: 0.75rem; }
.brand .name { font-size: 1.15rem; font-weight: 800; color: var(--ink); letter-spacing: -0.01em; }
.brand .sub { font-size: 0.8rem; color: var(--ink-soft); }
.pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 999px;
    padding: 0.35rem 0.85rem; font-size: 0.82rem; font-weight: 600; color: var(--ink-soft);
    box-shadow: var(--shadow-sm);
}
.pill .dot.ok { box-shadow: none; } .pill .dot.bad { box-shadow: none; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 22px;
    padding: 2.9rem 3rem 2.6rem 3rem; margin-bottom: 1.5rem;
    background:
        linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px) 0 0 / 44px 44px,
        linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px) 0 0 / 44px 44px,
        linear-gradient(135deg, #0F1B33 0%, #17315A 55%, #0C5E6B 100%);
    box-shadow: var(--shadow-md);
}
.hero .copy { position: relative; z-index: 2; max-width: 600px; }
.hero h1 {
    font-family: 'Newsreader', Georgia, serif; font-weight: 500;
    font-size: 2.9rem; line-height: 1.1; letter-spacing: -0.015em;
    color: #FFFFFF; margin: 0 0 0.9rem 0; padding: 0;
}
.hero p { color: #BCCDE8; font-size: 1.04rem; line-height: 1.65; margin: 0 0 1.4rem 0; }
.hero .badges { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.hero .badge {
    font-size: 0.8rem; font-weight: 600; color: #DDE8F7;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16);
    border-radius: 999px; padding: 0.3rem 0.8rem;
}
.hero .chain { position: absolute; right: 2.2rem; top: 50%; transform: translateY(-50%); z-index: 1; opacity: 0.95; }
@media (max-width: 1000px) { .hero .chain { display: none; } }
@media (max-width: 600px) { .hero { padding: 2rem 1.5rem; } .hero h1 { font-size: 2.1rem; } }

/* ---------- Search bar ---------- */
.st-key-searchbar {
    background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
    padding: 1.1rem 1.2rem 0.4rem 1.2rem; box-shadow: var(--shadow-md);
}
.st-key-searchbar .lead { font-size: 1.05rem; font-weight: 800; color: var(--ink); margin: 0 0 0.15rem 0; }
.st-key-searchbar .hint { font-size: 0.88rem; color: var(--ink-soft); margin: 0 0 0.8rem 0; }
.stTextInput input {
    background: #F8FAFC !important; color: var(--ink) !important;
    border: 1.5px solid var(--line) !important; border-radius: 12px !important;
    font-size: 1.03rem !important; padding: 0.8rem 1rem !important; min-height: 3.2rem;
}
.stTextInput input::placeholder { color: var(--ink-faint) !important; }
.stTextInput input:focus {
    background: #FFFFFF !important; border-color: var(--teal) !important;
    box-shadow: 0 0 0 4px rgba(14,124,134,0.13) !important;
}

/* ---------- Buttons ---------- */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #11909B 0%, #0E7C86 100%); border: none; color: #FFFFFF;
    font-weight: 700; border-radius: 12px; min-height: 3.2rem; font-size: 1rem;
    box-shadow: 0 6px 14px -6px rgba(14,124,134,0.65); transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #0E7C86 0%, #0B646D 100%); color: #FFFFFF;
    transform: translateY(-1px); box-shadow: 0 10px 18px -8px rgba(14,124,134,0.7);
}
.stApp .stButton > button:not([kind="primary"]), .stDownloadButton > button {
    background: var(--surface); color: var(--ink); border: 1px solid var(--line-strong);
    border-radius: 12px; font-weight: 600; font-size: 0.88rem; box-shadow: var(--shadow-sm);
}
.stApp .stButton > button:not([kind="primary"]):hover, .stDownloadButton > button:hover {
    border-color: var(--teal); color: var(--teal); background: #FAFEFE;
}
.st-key-chips .stButton > button { border-radius: 999px; min-height: 2.2rem; padding: 0.2rem 0.9rem; font-size: 0.82rem; }

/* ---------- Pipeline tracker ---------- */
.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.1rem; margin: 0.4rem 0 1.4rem 0; }
.stage {
    position: relative; background: var(--surface); border: 1.5px solid var(--line);
    border-radius: 16px; padding: 1.1rem 1.2rem 1.2rem 1.2rem; box-shadow: var(--shadow-sm);
    transition: border-color .2s ease, background .2s ease;
}
.stage:not(:last-child)::after {
    content: ""; position: absolute; top: 2.1rem; right: -1.1rem; width: 1.1rem; height: 2px;
    background: repeating-linear-gradient(90deg, var(--line-strong) 0 4px, transparent 4px 8px);
}
.stage .head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; }
.stage .glyph {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: #EEF2F8; color: var(--ink-soft);
}
.stage .name { font-size: 1.02rem; font-weight: 800; color: var(--ink); margin-bottom: 0.25rem; }
.stage .blurb { font-size: 0.84rem; color: var(--ink-soft); line-height: 1.5; min-height: 3em; }
.stage .state { margin-top: 0.9rem; font-size: 0.83rem; font-weight: 700; color: var(--ink-faint); }
.stage .bar { height: 3px; border-radius: 3px; background: var(--line); margin-top: 0.7rem; overflow: hidden; }

.stage .badge-ring { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; }
.stage.idle .badge-ring, .stage.pending .badge-ring { border: 2px solid var(--line-strong); }
.stage.running { border-color: var(--teal); background: #F5FBFB; }
.stage.running .glyph { background: var(--teal); color: #fff; }
.stage.running .state { color: var(--teal); }
.stage.running .badge-ring { border: 2.5px solid #BFE4E7; border-top-color: var(--teal); animation: spin 0.9s linear infinite; }
.stage.running .bar::after {
    content: ""; display: block; height: 100%; width: 40%; background: var(--teal);
    border-radius: 3px; animation: slide 1.3s ease-in-out infinite;
}
.stage.done { border-color: #BFE3CE; background: #F7FBF9; }
.stage.done .glyph { background: var(--green-soft); color: var(--green); }
.stage.done .state { color: var(--green); }
.stage.done .badge-ring { background: var(--green); color: #fff; }
.stage.done .bar { background: var(--green); }
.stage.error { border-color: #EFB5BF; background: #FEF6F7; }
.stage.error .glyph { background: var(--rose-soft); color: var(--rose); }
.stage.error .state { color: var(--rose); }
.stage.error .badge-ring { background: var(--rose); color: #fff; }
.stage.error .bar { background: var(--rose); }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes slide { 0% { margin-left: -40%; } 100% { margin-left: 100%; } }
@media (max-width: 960px) {
    .pipeline { grid-template-columns: repeat(2, 1fr); }
    .stage::after { display: none; }
}
@media (max-width: 540px) { .pipeline { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) {
    .stage.running .bar::after { animation: none; width: 100%; }
    .stage.running .badge-ring { animation: none; }
}

/* ---------- Feature cards ---------- */
.features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; margin-top: 0.2rem; }
.feature { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.2rem 1.3rem; }
.feature h4 { margin: 0 0 0.35rem 0; font-size: 0.98rem; font-weight: 800; color: var(--ink); }
.feature p { margin: 0; font-size: 0.88rem; line-height: 1.55; color: var(--ink-soft); }
@media (max-width: 800px) { .features { grid-template-columns: 1fr; } }

/* ---------- Results header & metrics ---------- */
.result-title {
    font-family: 'Newsreader', Georgia, serif; font-size: 2rem; font-weight: 500;
    line-height: 1.2; color: var(--ink); margin: 0.6rem 0 0.4rem 0; letter-spacing: -0.01em;
}
.result-meta { font-size: 0.88rem; color: var(--ink-soft); margin-bottom: 1.1rem; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.1rem; margin: 0.2rem 0 1.5rem 0; }
.metric {
    background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
    padding: 1.05rem 1.25rem; border-top: 3px solid var(--accent, var(--teal)); box-shadow: var(--shadow-sm);
}
.metric .value { font-size: 1.85rem; font-weight: 800; color: var(--ink); line-height: 1.1; letter-spacing: -0.02em; }
.metric .label { font-size: 0.84rem; color: var(--ink-soft); margin-top: 0.35rem; }
@media (max-width: 900px) { .metrics { grid-template-columns: repeat(2, 1fr); } }

/* ---------- Report ---------- */
.st-key-report_shell {
    background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
    padding: 2.2rem 2.7rem; box-shadow: var(--shadow-sm);
}
.st-key-report_shell h1, .st-key-report_shell h2, .st-key-report_shell h3 {
    font-family: 'Newsreader', Georgia, serif; font-weight: 600; color: var(--ink); letter-spacing: -0.01em;
}
.st-key-report_shell p, .st-key-report_shell li {
    font-family: 'Newsreader', Georgia, serif; font-size: 1.12rem; line-height: 1.78; color: #222E49;
}
.st-key-report_shell a { color: var(--teal); }
.outline { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem 1.25rem; }
.outline h4 { margin: 0 0 0.7rem 0; font-size: 0.95rem; font-weight: 800; color: var(--ink); }
.outline ul { list-style: none; margin: 0; padding: 0; }
.outline li {
    font-size: 0.86rem; color: var(--ink-soft); line-height: 1.4; padding: 0.45rem 0 0.45rem 0.8rem;
    border-left: 2px solid var(--line); margin-bottom: 0.1rem;
}
.outline li.l1 { border-left-color: var(--teal); color: var(--ink); font-weight: 700; }
.outline li.l3 { padding-left: 1.5rem; font-size: 0.82rem; }
.outline .facts { margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid var(--line); font-size: 0.84rem; color: var(--ink-soft); line-height: 1.7; }

/* ---------- Critic ---------- */
.score-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: 18px;
    padding: 1.5rem 1.2rem; text-align: center; box-shadow: var(--shadow-sm);
}
.score-card .verdict { font-size: 1rem; font-weight: 800; color: var(--ink); margin-top: 0.8rem; }
.score-card .note { font-size: 0.84rem; color: var(--ink-soft); margin-top: 0.3rem; line-height: 1.5; }
.st-key-critic_shell {
    background: var(--amber-soft); border: 1px solid #F0D9A6; border-left: 5px solid var(--amber);
    border-radius: 14px; padding: 1.5rem 1.9rem;
}
.st-key-critic_shell p, .st-key-critic_shell li { color: #4A3811; line-height: 1.7; font-size: 0.98rem; }
.st-key-critic_shell h1, .st-key-critic_shell h2, .st-key-critic_shell h3 { color: #3A2B0A; }

/* ---------- Sources ---------- */
.source {
    display: flex; align-items: center; gap: 0.9rem; text-decoration: none !important;
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
    padding: 0.75rem 1rem; margin-bottom: 0.6rem; transition: border-color .15s ease, box-shadow .15s ease;
}
.source:hover { border-color: var(--teal); box-shadow: var(--shadow-sm); }
.source .mark {
    flex: 0 0 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
    background: var(--teal-soft); color: var(--teal); font-weight: 800; font-size: 1rem;
}
.source .text { min-width: 0; }
.source .domain { font-size: 0.95rem; font-weight: 700; color: var(--ink); }
.source .path { font-size: 0.8rem; color: var(--ink-faint); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- Timing ---------- */
.timeline { display: flex; height: 14px; border-radius: 8px; overflow: hidden; background: var(--line); margin: 0.4rem 0 1.2rem 0; }
.timeline span { display: block; height: 100%; }
.trow {
    display: flex; align-items: center; gap: 0.8rem; background: var(--surface);
    border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 1rem; margin-bottom: 0.5rem; font-size: 0.92rem;
}
.trow .sw { width: 12px; height: 12px; border-radius: 4px; }
.trow .nm { flex: 1; font-weight: 700; color: var(--ink); }
.trow .tm { color: var(--ink-soft); font-variant-numeric: tabular-nums; }

/* ---------- Errors ---------- */
.error-box {
    background: var(--rose-soft); border: 1px solid #EFB5BF; border-left: 5px solid var(--rose);
    border-radius: 14px; padding: 1.1rem 1.4rem; color: #5E1B28; margin: 1rem 0; line-height: 1.6;
}
.error-box b { color: var(--rose); }

/* ---------- Tabs & misc ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 700; color: var(--ink-soft); padding: 0.65rem 1.1rem; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--teal) !important; }
.section-title { font-size: 1.25rem; font-weight: 800; color: var(--ink); margin: 1.8rem 0 0.7rem 0; letter-spacing: -0.01em; }
.section-sub { font-size: 0.92rem; color: var(--ink-soft); margin: -0.2rem 0 1rem 0; }
.footnote { font-size: 0.82rem; color: var(--ink-faint); text-align: center; margin-top: 3rem; }
</style>
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """Static description of one pipeline stage."""

    key: str
    title: str
    blurb: str
    working_label: str
    color: str


STAGES: List[Stage] = [
    Stage("search", "Search agent", "Finds recent, reliable sources on your topic.", "Searching the web", "#0E7C86"),
    Stage("reader", "Reader agent", "Picks the best result and reads the full page.", "Reading the top source", "#4F5FD5"),
    Stage("writer", "Writer", "Turns the research into a structured report.", "Drafting the report", "#C77D0A"),
    Stage("critic", "Critic", "Reviews the draft and points out weaknesses.", "Reviewing the draft", "#B4344A"),
]

# (button label, full topic sent to the agents)
EXAMPLE_TOPICS: List[Tuple[str, str]] = [
    ("AI and software jobs", "Impact of generative AI on software engineering jobs"),
    ("Solid-state batteries", "Solid-state batteries: where the technology stands"),
    ("GLP-1 and food industry", "How GLP-1 drugs are changing the food industry"),
    ("Small language models", "Rise of small language models for on-device AI"),
]

STATE_KEYS = {
    "search": "search_results",
    "reader": "scraped_content",
    "writer": "report",
    "critic": "feedback",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_text(content: Any) -> str:
    """Normalise model output (str, list of blocks, or message object) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if hasattr(content, "content"):
        return to_text(content.content)
    return str(content)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def reading_minutes(text: str) -> int:
    return max(1, round(word_count(text) / 220))


def extract_urls(text: str) -> List[str]:
    """Unique URLs in order of first appearance."""
    found = re.findall(r"https?://[^\s\)\]\>\"'<,]+", text or "")
    seen: Dict[str, None] = {}
    for url in found:
        seen.setdefault(url.rstrip(".;:"), None)
    return list(seen.keys())


def extract_score(feedback: str) -> Optional[float]:
    """Find an 'x/10' style score in the critic's feedback."""
    match = re.search(r"(\d{1,2}(?:\.\d+)?)\s*/\s*10", feedback or "")
    if not match:
        return None
    value = float(match.group(1))
    return value if 0 <= value <= 10 else None


def extract_headings(markdown: str) -> List[Tuple[int, str]]:
    """Return (level, text) for each markdown heading, for the outline panel."""
    headings: List[Tuple[int, str]] = []
    for match in re.finditer(r"^\s{0,3}(#{1,3})\s+(.+?)\s*#*\s*$", markdown or "", re.M):
        text = re.sub(r"[*_`]", "", match.group(2)).strip()
        if text:
            headings.append((len(match.group(1)), text))
    return headings


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit] or "research").rstrip("-")


def split_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path = parsed.path.rstrip("/")
    if parsed.query:
        path += "?" + parsed.query
    return domain or url, path or "/"


def build_markdown_export(entry: Dict[str, Any]) -> str:
    """Combine every pipeline output into one shareable markdown document."""
    state = entry["state"]
    stamp = entry["timestamp"].strftime("%d %B %Y, %H:%M")
    return (
        f"# {entry['topic']}\n\n"
        f"_Generated {stamp} in {fmt_seconds(entry['total_seconds'])}_\n\n---\n\n"
        f"## Report\n\n{state.get('report', '')}\n\n---\n\n"
        f"## Critic review\n\n{state.get('feedback', '')}\n\n---\n\n"
        f"## Search results\n\n{state.get('search_results', '')}\n\n---\n\n"
        f"## Scraped content\n\n{state.get('scraped_content', '')}\n"
    )


def build_json_export(entry: Dict[str, Any]) -> str:
    payload = {
        "topic": entry["topic"],
        "generated_at": entry["timestamp"].isoformat(),
        "total_seconds": round(entry["total_seconds"], 2),
        "stage_seconds": {k: round(v, 2) for k, v in entry["durations"].items()},
        **entry["state"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Session state and callbacks
# ---------------------------------------------------------------------------
def init_session() -> None:
    defaults: Dict[str, Any] = {
        "topic_input": "",
        "history": [],          # finished runs, newest first
        "active": None,         # the run currently displayed
        "last_error": None,     # dict with stage, message, trace
        "snippet_chars": 800,   # how much search text the reader sees
        "show_raw": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_topic(topic: str) -> None:
    st.session_state["topic_input"] = topic


def load_history_entry(index: int) -> None:
    history = st.session_state["history"]
    if 0 <= index < len(history):
        st.session_state["active"] = history[index]
        st.session_state["topic_input"] = history[index]["topic"]
        st.session_state["last_error"] = None


def clear_history() -> None:
    st.session_state["history"] = []
    st.session_state["active"] = None
    st.session_state["last_error"] = None


def start_new_research() -> None:
    st.session_state["active"] = None
    st.session_state["last_error"] = None
    st.session_state["topic_input"] = ""


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Raised when a stage fails; carries the stage for display."""

    def __init__(self, stage: Stage, original: BaseException) -> None:
        super().__init__(str(original))
        self.stage = stage
        self.original = original


def run_stage(stage_key: str, topic: str, state: Dict[str, str], snippet_chars: int) -> str:
    """Run one stage. Mirrors the logic in pipeline.py."""
    if stage_key == "search":
        agent = build_search_agent()
        result = agent.invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        return to_text(result["messages"][-1].content)

    if stage_key == "reader":
        agent = build_reader_agent()
        result = agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:snippet_chars]}",
                    )
                ]
            }
        )
        return to_text(result["messages"][-1].content)

    if stage_key == "writer":
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        return to_text(writer_chain.invoke({"topic": topic, "research": research_combined}))

    if stage_key == "critic":
        return to_text(critic_chain.invoke({"report": state["report"]}))

    raise ValueError(f"Unknown stage: {stage_key}")


def execute_pipeline(
    topic: str,
    snippet_chars: int,
    on_update: Callable[[Dict[str, str], Dict[str, float]], None],
) -> Dict[str, Any]:
    """
    Run all four stages in order. `on_update(statuses, timings)` is called after
    every change so the UI can redraw. Raises PipelineError on failure.
    """
    statuses = {s.key: "pending" for s in STAGES}
    timings: Dict[str, float] = {}
    state: Dict[str, str] = {}
    started = time.perf_counter()

    for stage in STAGES:
        statuses[stage.key] = "running"
        on_update(statuses, timings)
        t0 = time.perf_counter()
        try:
            output = run_stage(stage.key, topic, state, snippet_chars)
        except Exception as exc:  # noqa: BLE001
            statuses[stage.key] = "error"
            timings[stage.key] = time.perf_counter() - t0
            on_update(statuses, timings)
            raise PipelineError(stage, exc) from exc
        timings[stage.key] = time.perf_counter() - t0
        state[STATE_KEYS[stage.key]] = output
        statuses[stage.key] = "done"
        on_update(statuses, timings)

    return {
        "topic": topic,
        "state": state,
        "durations": timings,
        "total_seconds": time.perf_counter() - started,
        "timestamp": datetime.now(),
    }


# ---------------------------------------------------------------------------
# Rendering: header, hero, tracker
# ---------------------------------------------------------------------------
def render_topbar() -> None:
    ready = AGENTS_IMPORT_ERROR is None
    status = (
        '<span class="dot ok"></span>Agents ready'
        if ready
        else '<span class="dot bad"></span>Setup needed'
    )
    st.markdown(
        h(
            f"""
            <div class="topbar">
              <div class="brand">{LOGO_SVG}
                <div><div class="name">Research Desk</div>
                <div class="sub">Multi-agent research assistant</div></div>
              </div>
              <div class="pill">{status}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def chain_graphic() -> str:
    """Decorative diagram of the four agents handing work along a line."""
    points = [(50, 150), (135, 70), (225, 135), (310, 50)]
    path = "M" + " L".join(f"{x} {y}" for x, y in points)
    nodes = "".join(
        f'<g transform="translate({x},{y})">'
        '<circle r="27" fill="rgba(255,255,255,0.10)" stroke="rgba(255,255,255,0.35)" stroke-width="1.2"/>'
        '<g transform="translate(-11,-11)" fill="none" stroke="#FFFFFF" stroke-width="1.7" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICON_PATHS[stage.key]}</g></g>'
        for (x, y), stage in zip(points, STAGES)
    )
    return (
        '<svg width="360" height="210" viewBox="0 0 360 210" aria-hidden="true">'
        f'<path d="{path}" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="1.5" stroke-dasharray="5 6"/>'
        f"{nodes}</svg>"
    )


def render_hero() -> None:
    badges = ["Live web search", "Reads full pages", "Critic review", "Export to Markdown and JSON"]
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(
        h(
            f"""
            <div class="hero">
              <div class="copy">
                <h1>Ask a question. Four agents do the research.</h1>
                <p>One agent searches the web, another reads the best source, a writer drafts
                the report and a critic reviews it. You watch each step as it happens.</p>
                <div class="badges">{badge_html}</div>
              </div>
              <div class="chain">{chain_graphic()}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def tracker_html(statuses: Dict[str, str], timings: Dict[str, float]) -> str:
    cards: List[str] = []
    for stage in STAGES:
        status = statuses.get(stage.key, "idle")
        if status == "running":
            label = f"{stage.working_label}..."
            ring = ""
        elif status == "done":
            label = f"Finished in {fmt_seconds(timings.get(stage.key, 0))}"
            ring = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
        elif status == "error":
            label = "Stopped with an error"
            ring = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>'
        elif status == "pending":
            label = "Waiting"
            ring = ""
        else:
            label = "Ready"
            ring = ""
        cards.append(
            f'<div class="stage {status}">'
            f'<div class="head"><div class="glyph">{icon(stage.key)}</div>'
            f'<div class="badge-ring">{ring}</div></div>'
            f'<div class="name">{html.escape(stage.title)}</div>'
            f'<div class="blurb">{html.escape(stage.blurb)}</div>'
            f'<div class="state">{html.escape(label)}</div>'
            f'<div class="bar"></div></div>'
        )
    return f'<div class="pipeline">{"".join(cards)}</div>'


def render_tracker(placeholder, statuses: Dict[str, str], timings: Dict[str, float]) -> None:
    placeholder.markdown(tracker_html(statuses, timings), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rendering: input area
# ---------------------------------------------------------------------------
def render_input_form() -> Tuple[str, bool]:
    with st.container(key="searchbar"):
        st.markdown(
            h(
                """
                <div class="lead">What do you want to research?</div>
                <div class="hint">Describe a topic in a sentence. A full run usually takes one to three minutes.</div>
                """
            ),
            unsafe_allow_html=True,
        )
        col_input, col_button = st.columns([5, 1.25], vertical_alignment="center")
        with col_input:
            topic = st.text_input(
                "Research topic",
                key="topic_input",
                placeholder="For example: how small language models are changing mobile apps",
                label_visibility="collapsed",
            )
        with col_button:
            clicked = st.button("Start research", type="primary", **STRETCH)

    with st.container(key="chips"):
        columns = st.columns(len(EXAMPLE_TOPICS))
        for index, (column, (label, full_topic)) in enumerate(zip(columns, EXAMPLE_TOPICS)):
            with column:
                st.button(label, key=f"chip_{index}", on_click=set_topic, args=(full_topic,), **STRETCH)
    return topic.strip(), clicked


def render_empty_state() -> None:
    st.markdown('<div class="section-title">How a run works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Each agent passes its work to the next one.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(tracker_html({}, {}), unsafe_allow_html=True)
    st.markdown(
        h(
            """
            <div class="features">
              <div class="feature"><h4>Sources you can check</h4>
                <p>Every link the agents found is listed with its domain, so you can verify claims yourself.</p></div>
              <div class="feature"><h4>An honest second opinion</h4>
                <p>A separate critic reads the draft and flags gaps before you rely on it.</p></div>
              <div class="feature"><h4>Ready to share</h4>
                <p>Download the full research as Markdown, the report as text, or the raw data as JSON.</p></div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_error(error: Dict[str, str]) -> None:
    st.markdown(
        h(
            f"""
            <div class="error-box"><b>The {html.escape(error['stage'])} stopped before finishing.</b><br>
            {html.escape(error['message'])}<br><br>
            Check the API keys in <code>.env</code> and your internet connection, then start the research again.</div>
            """
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Technical details"):
        st.code(error.get("trace", ""), language="python")


# ---------------------------------------------------------------------------
# Rendering: results
# ---------------------------------------------------------------------------
def render_metrics(entry: Dict[str, Any]) -> None:
    state = entry["state"]
    report = state.get("report", "")
    sources = extract_urls(state.get("search_results", ""))
    score = extract_score(state.get("feedback", ""))
    items = [
        (fmt_seconds(entry["total_seconds"]), "Total research time", "#0E7C86"),
        (f"{word_count(report):,}", f"Words in report, {reading_minutes(report)} min read", "#4F5FD5"),
        (str(len(sources)), "Sources found", "#C77D0A"),
        (f"{score:g}/10" if score is not None else "n/a", "Critic score", "#B4344A"),
    ]
    cells = "".join(
        f'<div class="metric" style="--accent:{color}"><div class="value">{html.escape(value)}</div>'
        f'<div class="label">{html.escape(label)}</div></div>'
        for value, label, color in items
    )
    st.markdown(f'<div class="metrics">{cells}</div>', unsafe_allow_html=True)


def render_report_tab(entry: Dict[str, Any]) -> None:
    report = entry["state"].get("report", "")
    headings = extract_headings(report)
    main, side = st.columns([3.2, 1], gap="large")

    with main:
        with st.container(key="report_shell"):
            st.markdown(report or "_The writer returned no content._")

    with side:
        items = "".join(
            f'<li class="l{min(level, 3)}">{html.escape(text)}</li>' for level, text in headings[:14]
        )
        outline = f"<ul>{items}</ul>" if items else '<div class="side-note" style="color:#4A5875">No headings in this report.</div>'
        facts = (
            f"{word_count(report):,} words<br>{reading_minutes(report)} minute read<br>"
            f"{len(extract_urls(report))} links in the text"
        )
        st.markdown(
            h(f'<div class="outline"><h4>In this report</h4>{outline}<div class="facts">{facts}</div></div>'),
            unsafe_allow_html=True,
        )


def score_ring(score: float) -> str:
    radius, circumference = 54, 2 * 3.14159265 * 54
    filled = circumference * (score / 10)
    color = "#2B8A57" if score >= 8 else "#C77D0A" if score >= 6 else "#B4344A"
    return (
        '<svg width="140" height="140" viewBox="0 0 140 140" role="img" '
        f'aria-label="Critic score {score:g} out of 10">'
        f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="#E8ECF3" stroke-width="12"/>'
        f'<circle cx="70" cy="70" r="{radius}" fill="none" stroke="{color}" stroke-width="12" '
        f'stroke-linecap="round" stroke-dasharray="{filled:.1f} {circumference:.1f}" '
        'transform="rotate(-90 70 70)"/>'
        f'<text x="70" y="68" text-anchor="middle" font-size="34" font-weight="800" fill="#0F1B33" '
        f'font-family="Manrope, sans-serif">{score:g}</text>'
        '<text x="70" y="90" text-anchor="middle" font-size="13" fill="#8591A8" '
        'font-family="Manrope, sans-serif">out of 10</text></svg>'
    )


def verdict_for(score: float) -> Tuple[str, str]:
    if score >= 8:
        return "Strong draft", "The critic found few problems. Skim the notes for small fixes."
    if score >= 6:
        return "Solid, with gaps", "Usable as a starting point. Read the notes before sharing."
    return "Needs revision", "The critic found significant issues. Treat the report with care."


def render_critic_tab(entry: Dict[str, Any]) -> None:
    feedback = entry["state"].get("feedback", "")
    score = extract_score(feedback)

    def critic_body() -> None:
        with st.container(key="critic_shell"):
            st.markdown(feedback or "_The critic returned no feedback._")

    if score is None:
        critic_body()
        return

    verdict, note = verdict_for(score)
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown(
            h(
                f"""
                <div class="score-card">{score_ring(score)}
                <div class="verdict">{html.escape(verdict)}</div>
                <div class="note">{html.escape(note)}</div></div>
                """
            ),
            unsafe_allow_html=True,
        )
    with right:
        critic_body()


def render_sources_tab(entry: Dict[str, Any]) -> None:
    state = entry["state"]
    urls = list(
        dict.fromkeys(extract_urls(state.get("search_results", "")) + extract_urls(state.get("scraped_content", "")))
    )
    if not urls:
        st.info("No links were found in the agents' output.")
        return
    st.markdown(
        f'<div class="section-sub">{len(urls)} links found across searching and reading.</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for url in urls:
        domain, path = split_url(url)
        rows.append(
            f'<a class="source" href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'<div class="mark">{html.escape(domain[:1].upper() or "?")}</div>'
            f'<div class="text"><div class="domain">{html.escape(domain)}</div>'
            f'<div class="path">{html.escape(path)}</div></div></a>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_raw_tab(entry: Dict[str, Any]) -> None:
    state = entry["state"]
    with st.expander("Search agent output"):
        st.markdown(state.get("search_results", "") or "_Empty._")
    with st.expander("Reader agent output"):
        st.markdown(state.get("scraped_content", "") or "_Empty._")


def render_timing_tab(entry: Dict[str, Any]) -> None:
    total = entry["total_seconds"] or 1.0
    segments, rows = [], []
    for stage in STAGES:
        seconds = entry["durations"].get(stage.key, 0.0)
        share = seconds / total * 100
        segments.append(f'<span style="width:{share:.2f}%;background:{stage.color}"></span>')
        rows.append(
            f'<div class="trow"><span class="sw" style="background:{stage.color}"></span>'
            f'<span class="nm">{html.escape(stage.title)}</span>'
            f'<span class="tm">{fmt_seconds(seconds)} &nbsp;({share:.0f}%)</span></div>'
        )
    st.markdown(f'<div class="timeline">{"".join(segments)}</div>{"".join(rows)}', unsafe_allow_html=True)
    st.caption("Search and reading usually take longest because they wait on the network.")


def render_export_section(entry: Dict[str, Any]) -> None:
    slug = slugify(entry["topic"])
    st.markdown('<div class="section-title">Save this research</div>', unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    with left:
        st.download_button(
            "Download full document (.md)",
            data=build_markdown_export(entry),
            file_name=f"{slug}.md",
            mime="text/markdown",
            **STRETCH,
        )
    with middle:
        st.download_button(
            "Download report only (.txt)",
            data=entry["state"].get("report", ""),
            file_name=f"{slug}-report.txt",
            mime="text/plain",
            **STRETCH,
        )
    with right:
        st.download_button(
            "Download raw data (.json)",
            data=build_json_export(entry),
            file_name=f"{slug}.json",
            mime="application/json",
            **STRETCH,
        )


def render_results(entry: Dict[str, Any]) -> None:
    stamp = entry["timestamp"].strftime("%d %b %Y, %H:%M")
    head, action = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        st.markdown(
            h(
                f"""
                <div class="result-title">{html.escape(entry['topic'])}</div>
                <div class="result-meta">Researched on {html.escape(stamp)}</div>
                """
            ),
            unsafe_allow_html=True,
        )
    with action:
        st.button("New research", key="new_research", on_click=start_new_research, **STRETCH)

    render_metrics(entry)

    tab_names = ["Report", "Critic review", "Sources"]
    if st.session_state["show_raw"]:
        tab_names.append("Raw research")
    tab_names.append("Timing")
    lookup = dict(zip(tab_names, st.tabs(tab_names)))

    with lookup["Report"]:
        render_report_tab(entry)
    with lookup["Critic review"]:
        render_critic_tab(entry)
    with lookup["Sources"]:
        render_sources_tab(entry)
    if "Raw research" in lookup:
        with lookup["Raw research"]:
            render_raw_tab(entry)
    with lookup["Timing"]:
        render_timing_tab(entry)

    render_export_section(entry)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            h(
                f"""
                <div class="side-brand">{LOGO_SVG}
                <div><div class="t">Research Desk</div><div class="s">Settings and history</div></div></div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-label">Settings</div>', unsafe_allow_html=True)
        st.session_state["snippet_chars"] = st.slider(
            "Search text given to the reader (characters)",
            min_value=300,
            max_value=3000,
            value=st.session_state["snippet_chars"],
            step=100,
            help="pipeline.py passes the first 800 characters. Raise this if the reader picks poor links.",
        )
        st.session_state["show_raw"] = st.toggle(
            "Show raw agent output",
            value=st.session_state["show_raw"],
            help="Adds a tab with the unedited search and reader output.",
        )

        st.markdown('<div class="side-label">Recent research</div>', unsafe_allow_html=True)
        history = st.session_state["history"]
        if not history:
            st.markdown('<div class="side-note">Finished runs will appear here.</div>', unsafe_allow_html=True)
        for index, item in enumerate(history[:8]):
            label = item["topic"] if len(item["topic"]) <= 36 else item["topic"][:34] + "..."
            st.button(label, key=f"history_{index}", on_click=load_history_entry, args=(index,), **STRETCH)
        if history:
            st.button("Clear history", key="clear_history", on_click=clear_history, **STRETCH)

        st.markdown('<div class="side-label">Status</div>', unsafe_allow_html=True)
        if AGENTS_IMPORT_ERROR is None:
            st.markdown('<div class="side-status"><span class="dot ok"></span>Agents loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="side-status"><span class="dot bad"></span>Agents failed to load</div>'
                '<div class="side-note" style="margin-top:0.4rem">See the message on the main page.</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def handle_run(topic: str) -> None:
    """Validate input, run the pipeline with live progress, store the result."""
    st.session_state["last_error"] = None

    if AGENTS_IMPORT_ERROR is not None:
        st.session_state["last_error"] = {
            "stage": "agents module",
            "message": f"Could not import from agents.py: {AGENTS_IMPORT_ERROR}",
            "trace": "".join(traceback.format_exception(AGENTS_IMPORT_ERROR)),
        }
        return

    if len(topic) < 3:
        st.warning("Enter a topic of at least a few words so the agents have something to search for.")
        return

    st.markdown('<div class="section-title">Research in progress</div>', unsafe_allow_html=True)
    tracker_slot = st.empty()
    progress_slot = st.empty()

    def on_update(statuses: Dict[str, str], timings: Dict[str, float]) -> None:
        render_tracker(tracker_slot, statuses, timings)
        finished = sum(1 for s in statuses.values() if s == "done")
        progress_slot.progress(finished / len(STAGES))

    try:
        entry = execute_pipeline(topic, st.session_state["snippet_chars"], on_update)
    except PipelineError as err:
        st.session_state["last_error"] = {
            "stage": err.stage.title.lower(),
            "message": str(err.original) or err.original.__class__.__name__,
            "trace": "".join(traceback.format_exception(err.original)),
        }
        progress_slot.empty()
        return

    progress_slot.empty()
    tracker_slot.empty()
    st.session_state["history"].insert(0, entry)
    st.session_state["active"] = entry


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()
    render_sidebar()
    render_topbar()
    render_hero()

    topic, clicked = render_input_form()
    if clicked:
        handle_run(topic)

    if st.session_state["last_error"]:
        render_error(st.session_state["last_error"])

    active = st.session_state["active"]
    if active is not None:
        render_results(active)
    elif not st.session_state["last_error"]:
        render_empty_state()

    st.markdown(
        '<div class="footnote">Reports are written by AI from live web sources. '
        "Check important facts before relying on them.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()