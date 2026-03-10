"""app.py – Sentinel AI: Bento-Box + Dark Sentinel + AI-First Dashboard."""

from __future__ import annotations

import html
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import (
    RedditConfig,
    analyze_post,
    extract_keywords,
    fetch_reddit_posts,
    load_models,
)
from utils import generate_ai_response, read_stream_source, summarize_root_cause

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# MEGA CSS — Dark Sentinel + Bento Grid + Glassmorphism + Animations
# ---------------------------------------------------------------------------
SENTINEL_CSS = """
<style>
/* ======= DESIGN TOKENS ======= */
:root {
    --bg-base: #121212;
    --bg-surface: #1a1a2e;
    --bg-card: rgba(26,26,46,0.65);
    --border-card: rgba(80,120,200,0.18);
    --text-primary: #e8eaed;
    --text-secondary: #8892a4;
    --accent-blue: #00d4ff;
    --accent-blue-glow: rgba(0,212,255,0.35);
    --accent-red: #ff2d55;
    --accent-red-glow: rgba(255,45,85,0.30);
    --accent-green: #00e676;
    --accent-green-glow: rgba(0,230,118,0.25);
    --accent-amber: #ffab00;
    --glass-bg: rgba(26,26,46,0.55);
    --glass-border: rgba(255,255,255,0.08);
    --glass-blur: blur(16px);
    --radius: 16px;
    --radius-sm: 10px;
    --radius-pill: 999px;
}

/* ======= GLOBAL ======= */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #121212 40%, #0d1b2a 100%) !important;
    color: var(--text-primary);
}
.block-container { padding-top: 0.8rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ======= GLASSMORPHISM CARD ======= */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,212,255,0.10);
}

/* ======= HEADER BAR ======= */
.sentinel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: var(--glass-bg);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    margin-bottom: 16px;
}
.sentinel-title {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-blue), #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.status-dot {
    width: 12px; height: 12px; border-radius: 50%;
    display: inline-block; margin-right: 8px;
    animation: pulse 2s infinite;
}
.status-green  { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green-glow); }
.status-amber  { background: var(--accent-amber); box-shadow: 0 0 8px rgba(255,171,0,0.4); }
.status-red    { background: var(--accent-red);   box-shadow: 0 0 8px var(--accent-red-glow); }

/* ======= PILL BADGES ======= */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: var(--radius-pill);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.pill-pos  { background: rgba(0,230,118,0.15); color: var(--accent-green); border: 1px solid rgba(0,230,118,0.3); }
.pill-neg  { background: rgba(255,45,85,0.15);  color: var(--accent-red);   border: 1px solid rgba(255,45,85,0.3); }
.pill-neu  { background: rgba(136,146,164,0.15); color: var(--text-secondary); border: 1px solid rgba(136,146,164,0.3); }
.pill-sarc { background: rgba(255,171,0,0.15);  color: var(--accent-amber); border: 1px solid rgba(255,171,0,0.3); }

/* ======= KPI METRIC CARDS ======= */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ======= FEED ITEMS ======= */
.feed-card {
    background: var(--glass-bg);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--glass-border);
    border-left: 4px solid transparent;
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    transition: background 0.2s;
}
.feed-card:hover { background: rgba(26,26,46,0.85); }
.feed-pos  { border-left-color: var(--accent-green); }
.feed-neg  { border-left-color: var(--accent-red); }
.feed-neu  { border-left-color: var(--text-secondary); }
.feed-sarc { border-left-color: var(--accent-amber); }
.feed-meta { color: var(--text-secondary); font-size: 0.78rem; margin-top: 4px; }

/* ======= KEYWORD BAR ======= */
.kw-bar-wrap {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px; font-size: 0.85rem;
}
.kw-label { min-width: 80px; font-weight: 600; }
.kw-track { flex: 1; height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden; }
.kw-fill  { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
.kw-fill-neg { background: linear-gradient(90deg, var(--accent-red), #ff6b6b); }
.kw-fill-pos { background: linear-gradient(90deg, var(--accent-green), #69f0ae); }
.kw-count { color: var(--text-secondary); font-size: 0.78rem; min-width: 24px; text-align: right; }

/* ======= SECTION TITLE ======= */
.brick-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* ======= AI INSIGHT BUBBLE ======= */
.ai-bubble {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,77,255,0.08));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.5;
}
.ai-bubble-label {
    font-size: 0.72rem; font-weight: 700; color: var(--accent-blue);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;
}

/* ======= GLOW BUTTON ======= */
.glow-btn {
    display: inline-block;
    padding: 10px 28px;
    background: linear-gradient(135deg, var(--accent-blue), #7c4dff);
    color: #fff !important;
    font-weight: 700;
    font-size: 0.9rem;
    border: none;
    border-radius: var(--radius-pill);
    cursor: pointer;
    box-shadow: 0 0 20px var(--accent-blue-glow);
    transition: box-shadow 0.3s, transform 0.2s;
    text-decoration: none;
}
.glow-btn:hover {
    box-shadow: 0 0 30px var(--accent-blue-glow), 0 0 60px rgba(124,77,255,0.2);
    transform: scale(1.04);
}

/* ======= ANIMATIONS ======= */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: slideUp 0.4s ease-out; }

/* ======= SARCASM BADGE ======= */
.sarcasm-icon { font-size: 0.9rem; margin-right: 2px; }

/* ======= DEEP-DIVE PANEL ======= */
.deep-dive-header {
    font-size: 0.88rem; font-weight: 600;
    color: var(--accent-amber);
    padding: 8px 0;
}

/* ======= SCROLLBAR ======= */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
"""
st.markdown(SENTINEL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def get_models():
    return load_models()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
HISTORY_COLS = [
    "timestamp", "raw_text", "clean_text", "aspect",
    "base_sentiment", "base_score", "sentiment_tag",
    "final_sentiment", "is_sarcastic", "followers",
    "crisis_score", "source",
]


def init_state() -> None:
    if "history_df" not in st.session_state:
        st.session_state.history_df = pd.DataFrame(columns=HISTORY_COLS)
    if "ai_chat" not in st.session_state:
        st.session_state.ai_chat = []


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
def current_crisis_gauge(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(float(df["crisis_score"].tail(20).mean()), 2)


def net_sentiment_pct(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    pos = int((df["final_sentiment"] == "Positive").sum())
    neg = int((df["final_sentiment"] == "Negative").sum())
    return round((pos - neg) / max(len(df), 1) * 100, 2)


def _crisis_status(gauge: float) -> tuple:
    """Return (css_class, label) for the status dot."""
    if gauge < 30:
        return "status-green", "ALL CLEAR"
    if gauge < 60:
        return "status-amber", "ELEVATED"
    return "status-red", "CRISIS"


def _sentiment_pill(tag: str) -> str:
    low = tag.lower()
    if "sarcas" in low:
        return f'<span class="pill pill-sarc"><span class="sarcasm-icon">🕵️</span> {html.escape(tag)}</span>'
    if "positive" in low:
        return f'<span class="pill pill-pos">+ {html.escape(tag)}</span>'
    if "negative" in low:
        return f'<span class="pill pill-neg">− {html.escape(tag)}</span>'
    return f'<span class="pill pill-neu">● {html.escape(tag)}</span>'


# ---------------------------------------------------------------------------
# HEADER with crisis status lamp
# ---------------------------------------------------------------------------
def render_header(df: pd.DataFrame) -> None:
    gauge_val = current_crisis_gauge(df)
    status_css, status_label = _crisis_status(gauge_val)
    count = len(df)
    is_live = st.session_state.get("stream_active", False)
    live_badge = (
        '<span style="display:flex;align-items:center;gap:4px;'
        'background:rgba(0,230,118,0.15);border:1px solid rgba(0,230,118,0.3);'
        'border-radius:999px;padding:3px 12px;font-size:0.75rem;font-weight:700;'
        'color:#00e676;letter-spacing:1px;">'
        '<span class="status-dot status-green" style="width:8px;height:8px;"></span>'
        'LIVE</span>'
    ) if is_live else (
        '<span style="font-size:0.75rem;color:var(--text-secondary);'
        'letter-spacing:1px;">IDLE</span>'
    )
    st.markdown(
        f"""<div class="sentinel-header animate-in">
            <span class="sentinel-title">🛡️ Sentinel AI</span>
            <span style="display:flex;align-items:center;gap:16px;">
                {live_badge}
                <span style="font-size:0.82rem;color:var(--text-secondary);">{count} mentions analysed</span>
                <span style="display:flex;align-items:center;">
                    <span class="status-dot {status_css}"></span>
                    <span style="font-size:0.82rem;font-weight:700;letter-spacing:1px;">{status_label}</span>
                </span>
            </span>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# BENTO ROW 1: KPI cards
# ---------------------------------------------------------------------------
def render_kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mentions", len(df))
    c2.metric("Net Sentiment", f"{net_sentiment_pct(df):+.1f}%")
    avg_crisis = round(float(df["crisis_score"].mean()), 1) if not df.empty else 0.0
    c3.metric("Crisis Score (avg)", avg_crisis)
    sarc_count = int(df["is_sarcastic"].sum()) if not df.empty and "is_sarcastic" in df.columns else 0
    c4.metric("🕵️ Sarcasm Detected", sarc_count)


# ---------------------------------------------------------------------------
# BENTO ROW 2: Gauge + Sentiment Area + Source Donut
# ---------------------------------------------------------------------------
def render_bento_row2(df: pd.DataFrame) -> None:
    col_gauge, col_area, col_donut = st.columns([1, 2, 1])

    # --- Crisis Speedometer ---
    with col_gauge:
        st.markdown('<div class="brick-title">Crisis Gauge</div>', unsafe_allow_html=True)
        gauge_val = current_crisis_gauge(df)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={"font": {"size": 42, "color": "#e8eaed"}, "suffix": ""},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892a4", "tickfont": {"color": "#8892a4"}},
                "bar": {"color": "#00d4ff", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,230,118,0.12)"},
                    {"range": [30, 60], "color": "rgba(255,171,0,0.12)"},
                    {"range": [60, 100], "color": "rgba(255,45,85,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ff2d55", "width": 3},
                    "thickness": 0.85,
                    "value": 80,
                },
            },
        ))
        fig.update_layout(
            height=240, margin=dict(t=30, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaed",
        )
        st.plotly_chart(fig, width="stretch", key="crisis_gauge_main")

    # --- Sentiment Over Time (glowing area chart) ---
    with col_area:
        st.markdown('<div class="brick-title">Sentiment Trend</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("Waiting for stream data…")
        else:
            trend = df.copy()
            trend["minute"] = trend["timestamp"].dt.floor("min")
            grouped = trend.groupby(["minute", "final_sentiment"]).size().reset_index(name="count")
            grouped = grouped[grouped["final_sentiment"].isin(["Positive", "Negative"])]
            if grouped.empty:
                st.info("Not enough polarity data yet.")
            else:
                fig = go.Figure()
                for sent, color, fill_color in [
                    ("Positive", "#00e676", "rgba(0,230,118,0.15)"),
                    ("Negative", "#ff2d55", "rgba(255,45,85,0.15)"),
                ]:
                    sub = grouped[grouped["final_sentiment"] == sent].sort_values("minute")
                    if not sub.empty:
                        fig.add_trace(go.Scatter(
                            x=sub["minute"], y=sub["count"], name=sent,
                            mode="lines", line=dict(color=color, width=2.5),
                            fill="tozeroy", fillcolor=fill_color,
                        ))
                fig.update_layout(
                    height=240, margin=dict(t=10, b=30, l=40, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8eaed", legend=dict(orientation="h", y=-0.15),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
                )
                st.plotly_chart(fig, width="stretch", key="sentiment_area")

    # --- Source Donut ---
    with col_donut:
        st.markdown('<div class="brick-title">Source Split</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("No data.")
        else:
            src = df["source"].value_counts().reset_index()
            src.columns = ["source", "count"]
            fig = go.Figure(go.Pie(
                labels=src["source"], values=src["count"],
                hole=0.6, textinfo="percent+label",
                marker=dict(colors=["#00d4ff", "#7c4dff", "#ff2d55", "#ffab00"]),
            ))
            fig.update_layout(
                height=240, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaed", showlegend=False,
            )
            st.plotly_chart(fig, width="stretch", key="source_donut")


# ---------------------------------------------------------------------------
# BENTO ROW 3: Topic Breakdown + Top Keywords
# ---------------------------------------------------------------------------
def render_bento_row3(df: pd.DataFrame) -> None:
    col_topic, col_kw = st.columns([3, 2])

    # --- Topic Breakdown (horizontal grouped bar) ---
    with col_topic:
        st.markdown('<div class="brick-title">Aspect Breakdown</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("Topic data appears after analysis.")
        else:
            topic = df.groupby(["aspect", "final_sentiment"]).size().reset_index(name="count")
            topic = topic[topic["final_sentiment"].isin(["Positive", "Negative", "Informational"])]
            if topic.empty:
                st.info("Not enough data.")
            else:
                totals = topic.groupby("aspect")["count"].transform("sum")
                topic["percent"] = (topic["count"] / totals * 100).round(1)
                fig = px.bar(
                    topic, y="aspect", x="percent", color="final_sentiment",
                    orientation="h", barmode="group",
                    color_discrete_map={
                        "Positive": "#00e676", "Negative": "#ff2d55", "Informational": "#8892a4",
                    },
                )
                fig.update_layout(
                    height=260, margin=dict(t=10, b=30, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8eaed", xaxis_title="% of Aspect",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, width="stretch", key="topic_bars")

    # --- Top Keywords with sentiment meters ---
    with col_kw:
        st.markdown('<div class="brick-title">Top Keywords</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("Keywords appear after analysis.")
        else:
            kw_data = extract_keywords(df, top_n=8)
            if not kw_data:
                st.info("Not enough text data.")
            else:
                max_count = max(k["count"] for k in kw_data) or 1
                bars_html = ""
                for kw in kw_data:
                    pct = int(kw["count"] / max_count * 100)
                    fill_cls = "kw-fill-neg" if kw["sentiment"] == "Negative" else "kw-fill-pos"
                    bars_html += (
                        f'<div class="kw-bar-wrap">'
                        f'<span class="kw-label">{html.escape(kw["keyword"])}</span>'
                        f'<div class="kw-track"><div class="kw-fill {fill_cls}" '
                        f'style="width:{pct}%"></div></div>'
                        f'<span class="kw-count">{kw["count"]}</span>'
                        f'</div>'
                    )
                st.markdown(f'<div class="glass-card animate-in">{bars_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# BENTO ROW 4: Live Feed (glassmorphism wide brick)
# ---------------------------------------------------------------------------
def render_live_feed(df: pd.DataFrame) -> None:
    st.markdown('<div class="brick-title">Live Feed</div>', unsafe_allow_html=True)
    feed = df.tail(12).iloc[::-1] if not df.empty else pd.DataFrame()
    if feed.empty:
        st.info("Feed idle — start a stream or fetch Reddit data.")
        return

    cols = st.columns(3)
    for idx, (_, row) in enumerate(feed.iterrows()):
        tag = str(row["sentiment_tag"]).lower()
        css = "feed-neu"
        if "positive" in tag:
            css = "feed-pos"
        elif "negative" in tag or "sarcas" in tag:
            css = "feed-neg" if "negative" in tag else "feed-sarc"
        safe_text = html.escape(str(row["raw_text"])[:180])
        pill = _sentiment_pill(str(row["sentiment_tag"]))
        sarc_badge = '<span class="sarcasm-icon">🕵️</span> ' if row.get("is_sarcastic") else ""
        card_html = (
            f'<div class="feed-card {css} animate-in">'
            f'{sarc_badge}{pill} &nbsp; '
            f'<span class="pill pill-neu">{html.escape(str(row["aspect"]))}</span> &nbsp; '
            f'<span style="color:var(--accent-blue);font-weight:600;">⚡ {row["crisis_score"]}</span>'
            f'<div style="margin-top:6px;">{safe_text}</div>'
            f'<div class="feed-meta">{row["timestamp"]} · {row["source"]} · {int(row["followers"])} followers</div>'
            f'</div>'
        )
        cols[idx % 3].markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DEEP DIVE expandable panel
# ---------------------------------------------------------------------------
def render_deep_dive(df: pd.DataFrame) -> None:
    if df.empty:
        return
    with st.expander("🔍 Deep Dive — Spike & Anomaly Analysis"):
        if "crisis_score" not in df.columns:
            st.info("No crisis data available.")
            return
        high = df[df["crisis_score"] > 70].sort_values("crisis_score", ascending=False).head(5)
        if high.empty:
            st.success("No high-crisis spikes detected (all scores < 70).")
            return
        st.markdown('<div class="deep-dive-header">Highest Crisis Posts</div>', unsafe_allow_html=True)
        for _, row in high.iterrows():
            pill = _sentiment_pill(str(row["sentiment_tag"]))
            sarc = " 🕵️ SARCASM" if row.get("is_sarcastic") else ""
            st.markdown(
                f'{pill}{sarc} &nbsp; **Crisis: {row["crisis_score"]}** '
                f'| {int(row["followers"])} followers | {row["aspect"]}<br/>'
                f'<span style="color:var(--text-secondary);font-size:0.88rem;">'
                f'{html.escape(str(row["raw_text"])[:200])}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("---")


# ---------------------------------------------------------------------------
# AI SIDEBAR (right 30% — conversational insights + response studio)
# ---------------------------------------------------------------------------
def render_ai_sidebar(df: pd.DataFrame) -> None:
    st.markdown('<div class="brick-title">🤖 AI Copilot</div>', unsafe_allow_html=True)

    # --- Auto-insights ---
    if df.empty:
        st.markdown(
            '<div class="ai-bubble"><div class="ai-bubble-label">Sentinel AI</div>'
            "Loading analysis data… dashboard will populate momentarily.</div>",
            unsafe_allow_html=True,
        )
        return

    gauge_val = current_crisis_gauge(df)
    net = net_sentiment_pct(df)
    sarc_count = int(df["is_sarcastic"].sum()) if "is_sarcastic" in df.columns else 0
    neg_count = int((df["final_sentiment"] == "Negative").sum())
    total = len(df)

    # Situational insight
    if gauge_val >= 60:
        insight = (
            f"⚠️ <b>Crisis Alert!</b> Average crisis score is <b>{gauge_val}</b>. "
            f"{neg_count}/{total} posts are negative. Immediate attention recommended."
        )
    elif gauge_val >= 30:
        insight = (
            f"🟡 Sentiment is mixed. Crisis score at <b>{gauge_val}</b>. "
            f"Net sentiment: <b>{net:+.1f}%</b>. Monitor closely."
        )
    else:
        insight = (
            f"✅ Sentiment looks healthy! Crisis score is only <b>{gauge_val}</b>. "
            f"Net sentiment: <b>{net:+.1f}%</b>."
        )
    if sarc_count > 0:
        insight += f" 🕵️ Detected <b>{sarc_count}</b> sarcastic post(s) — flagged for review."

    st.markdown(
        f'<div class="ai-bubble animate-in"><div class="ai-bubble-label">Sentinel AI</div>{insight}</div>',
        unsafe_allow_html=True,
    )

    # Root-cause insight
    negatives = df[df["final_sentiment"] == "Negative"]["raw_text"].tolist()
    if negatives:
        root_cause = summarize_root_cause(negatives)
        st.markdown(
            f'<div class="ai-bubble"><div class="ai-bubble-label">Root Cause</div>{html.escape(root_cause)}</div>',
            unsafe_allow_html=True,
        )

    # --- AI Response Studio ---
    st.markdown("---")
    st.markdown('<div class="brick-title">✍️ Response Studio</div>', unsafe_allow_html=True)
    crisis_posts = df[df["crisis_score"] > 80] if not df.empty else pd.DataFrame()

    if crisis_posts.empty:
        st.markdown(
            '<div class="ai-bubble"><div class="ai-bubble-label">Status</div>'
            "No posts exceed crisis threshold (80). Response drafting not needed.</div>",
            unsafe_allow_html=True,
        )
    else:
        worst = crisis_posts.sort_values("crisis_score", ascending=False).iloc[0]
        st.markdown(
            f'<div style="background:rgba(255,45,85,0.1);border:1px solid rgba(255,45,85,0.3);'
            f'border-radius:var(--radius);padding:12px;margin-bottom:10px;font-size:0.88rem;">'
            f'<b style="color:var(--accent-red);">Crisis Post (Score: {worst["crisis_score"]})</b><br/>'
            f'{html.escape(str(worst["raw_text"])[:250])}</div>',
            unsafe_allow_html=True,
        )
        root = summarize_root_cause(negatives) if negatives else ""
        if st.button("✨ Generate PR Response", key="gen_response_btn"):
            responses = generate_ai_response(str(worst["raw_text"]), root)
            st.session_state.ai_chat = [
                {"tone": tone, "text": text} for tone, text in responses.items()
            ]

        for msg in st.session_state.get("ai_chat", []):
            tone_colors = {
                "Professional": "var(--accent-blue)",
                "Empathic": "var(--accent-green)",
                "Brand-Witty": "var(--accent-amber)",
            }
            color = tone_colors.get(msg["tone"], "var(--text-secondary)")
            st.markdown(
                f'<div class="ai-bubble animate-in">'
                f'<div class="ai-bubble-label" style="color:{color};">{html.escape(msg["tone"])}</div>'
                f'{html.escape(msg["text"])}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Batch-analyse helper (processes all posts at once — no flickering)
# ---------------------------------------------------------------------------
DEFAULT_SOURCE = str(Path("data") / "sample_posts.csv")


def _analyse_all(source: str, models) -> pd.DataFrame:
    """Read source and analyse every post in one shot. Cached per source path."""
    posts = read_stream_source(source)
    rows = [analyze_post(p, models) for p in posts]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _ensure_initial_data() -> None:
    """On first load, batch-process the entire CSV so charts appear instantly."""
    if st.session_state.get("initial_load_done"):
        return
    st.session_state["initial_load_done"] = True
    try:
        models = get_models()
        all_posts = read_stream_source(DEFAULT_SOURCE)
        st.session_state["all_posts"] = all_posts

        progress = st.progress(0, text="Analysing posts…")
        total = len(all_posts)
        rows = []
        for i, p in enumerate(all_posts):
            rows.append(analyze_post(p, models))
            progress.progress((i + 1) / total, text=f"Analysing post {i + 1}/{total}…")
        progress.empty()

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        st.session_state.history_df = df
        st.session_state["stream_idx"] = len(df)
    except Exception as exc:
        st.error(f"Initial load failed: {exc}")


# ---------------------------------------------------------------------------
# Fragment: seamless live ticker (adds 1 post per interval, no page reload)
# ---------------------------------------------------------------------------
@st.fragment(run_every=timedelta(seconds=2))
def live_ticker() -> None:
    """Runs as an independent fragment — only this piece re-executes,
    so the rest of the page stays perfectly still (no flicker)."""
    models = get_models()
    posts = st.session_state.get("all_posts", [])
    if not posts or not st.session_state.get("stream_active", False):
        return

    idx = st.session_state.get("stream_idx", 0)
    # Wrap around for continuous loop
    actual_idx = idx % len(posts)
    post = posts[actual_idx]

    analyzed = analyze_post(post, models)
    new_row = pd.DataFrame([analyzed])
    frames = [st.session_state.history_df, new_row]
    frames = [f for f in frames if not f.empty and not f.isna().all(axis=None)]
    st.session_state.history_df = pd.concat(frames, ignore_index=True)
    st.session_state.history_df["timestamp"] = pd.to_datetime(
        st.session_state.history_df["timestamp"], errors="coerce"
    )
    st.session_state["stream_idx"] = idx + 1


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def sidebar_controls() -> None:
    st.sidebar.markdown("### ⚡ Live Analysis")

    source_file = st.sidebar.text_input("CSV / JSON path", value=DEFAULT_SOURCE)
    st.session_state["stream_file"] = source_file

    is_active = st.session_state.get("stream_active", True)
    idx = st.session_state.get("stream_idx", 0)
    total = len(st.session_state.get("all_posts", []))

    if is_active:
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:8px 0;">'
            f'<span class="status-dot status-green"></span>'
            f'<span style="font-size:0.85rem;font-weight:600;">LIVE — {idx} processed</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("⏹ Pause Live Feed"):
            st.session_state["stream_active"] = False
            st.rerun()
    else:
        st.sidebar.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:8px 0;">'
            '<span style="width:12px;height:12px;border-radius:50%;background:#8892a4;'
            'display:inline-block;"></span>'
            '<span style="font-size:0.85rem;color:var(--text-secondary);">Paused</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("▶ Resume Live Feed"):
            st.session_state["stream_active"] = True
            st.rerun()

    if st.sidebar.button("🔄 Reset & Reload"):
        st.session_state.history_df = pd.DataFrame(columns=HISTORY_COLS)
        st.session_state["initial_load_done"] = False
        st.session_state["stream_idx"] = 0
        st.session_state["stream_active"] = True
        st.session_state.ai_chat = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Reddit (PRAW)")
    r_client = st.sidebar.text_input("Client ID", type="password")
    r_secret = st.sidebar.text_input("Client Secret", type="password")
    r_sub = st.sidebar.text_input("Subreddit", value="technology")
    r_limit = st.sidebar.slider("Comment limit", 5, 100, 25)

    if st.sidebar.button("▶ Fetch Reddit Comments"):
        cfg = RedditConfig(
            client_id=r_client, client_secret=r_secret,
            subreddit=r_sub, limit=r_limit,
        )
        reddit_posts = fetch_reddit_posts(cfg)
        if not reddit_posts:
            st.sidebar.warning("No Reddit data returned.")
        else:
            models = get_models()
            rows = [analyze_post(p, models) for p in reddit_posts]
            new_df = pd.DataFrame(rows)
            frames = [st.session_state.history_df, new_df]
            frames = [f for f in frames if not f.empty and not f.isna().all(axis=None)]
            st.session_state.history_df = pd.concat(frames, ignore_index=True)
            st.session_state.history_df["timestamp"] = pd.to_datetime(
                st.session_state.history_df["timestamp"], errors="coerce"
            )
            st.sidebar.success(f"Analysed {len(reddit_posts)} Reddit comments.")


# ---------------------------------------------------------------------------
# MAIN LAYOUT — 70/30 split: Dashboard | AI Sidebar
# ---------------------------------------------------------------------------
def main() -> None:
    init_state()
    _ensure_initial_data()
    sidebar_controls()

    df = st.session_state.history_df
    render_header(df)

    main_col, ai_col = st.columns([7, 3], gap="medium")

    with main_col:
        render_kpi_row(df)
        render_bento_row2(df)
        render_bento_row3(df)
        render_live_feed(df)
        render_deep_dive(df)

    with ai_col:
        render_ai_sidebar(df)

    # Seamless live ticker — runs as an independent fragment,
    # adds 1 new post every 2 seconds WITHOUT refreshing the page
    live_ticker()


main()
