"""utils.py – Preprocessing, emoji mapping, text helpers, and data-source readers."""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, Iterable, List

import pandas as pd


# ---------------------------------------------------------------------------
# Emoji → text mapping (preserves emotional intent during NLP inference)
# ---------------------------------------------------------------------------
EMOJI_MAP: Dict[str, str] = {
    "🚀": " bullish excited ",
    "📉": " falling bad ",
    "😡": " angry furious ",
    "🔥": " trending intense ",
    "😐": " neutral ",
    "😊": " happy pleased ",
    "💀": " terrible dead ",
    "👍": " good approve ",
    "👎": " bad disapprove ",
    "❤️": " love ",
    "💔": " heartbroken disappointed ",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "is", "it",
    "this", "that", "on", "in", "we", "our", "your", "with", "from",
    "at", "are", "be", "was", "were", "has", "have", "had", "not",
    "but", "they", "them", "its", "just", "about", "been", "would",
}

NEGATIVE_CONTEXT_WORDS = {
    "garbage", "fail", "broken", "worst", "awful", "trash",
    "pathetic", "buggy", "terrible", "useless", "scam", "joke",
}


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Clean noise (RTs, URLs, mentions) but preserve mapped emoji intent."""
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    mapped = raw
    for emoji, phrase in EMOJI_MAP.items():
        mapped = mapped.replace(emoji, phrase)

    mapped = re.sub(r"\bRT\b", " ", mapped)
    mapped = re.sub(r"https?://\S+", " ", mapped)
    mapped = re.sub(r"@[A-Za-z0-9_]+", " ", mapped)
    mapped = re.sub(r"#", "", mapped)
    mapped = re.sub(r"[^\w\s!?.,'-]", " ", mapped)
    mapped = re.sub(r"\s+", " ", mapped).strip()
    return mapped


# ---------------------------------------------------------------------------
# Sarcasm detection heuristic
# ---------------------------------------------------------------------------
def detect_sarcasm(original_text: str, sentiment_label: str) -> bool:
    """Flag as sarcastic when model says Positive but text contains negative-context words."""
    if sentiment_label != "Positive":
        return False
    lower = original_text.lower()
    return any(word in lower for word in NEGATIVE_CONTEXT_WORDS)


# ---------------------------------------------------------------------------
# Root-cause summariser (extractive, no external LLM needed)
# ---------------------------------------------------------------------------
def summarize_root_cause(negative_mentions: Iterable[str]) -> str:
    mentions = [m for m in negative_mentions if m and m.strip()]
    if not mentions:
        return "No critical root cause detected from recent negative mentions."

    tokens: List[str] = []
    for mention in mentions[-50:]:
        clean = preprocess_text(mention).lower()
        words = [w for w in re.findall(r"[a-z']+", clean) if w not in STOPWORDS and len(w) > 2]
        tokens.extend(words)

    if not tokens:
        return "Recent negative mentions are mixed or context-light; manual review recommended."

    common = [word for word, _ in Counter(tokens).most_common(3)]
    return f"Recurring issues around: {', '.join(common)}."


# ---------------------------------------------------------------------------
# AI response generator (3 tones)
# ---------------------------------------------------------------------------
def generate_ai_response(post_text: str, root_cause: str) -> Dict[str, str]:
    """Draft PR-style replies in three tones for a high-crisis post."""
    snippet = (post_text[:120] + "...") if len(post_text) > 120 else post_text
    return {
        "Professional": (
            f'We appreciate you raising this. Regarding "{snippet}" — '
            f"our team is actively investigating. {root_cause} "
            "We will provide an update within 24 hours."
        ),
        "Empathic": (
            f"We hear you, and we're sorry about this experience. "
            f"{root_cause} Your feedback matters deeply, and we're "
            "fast-tracking a resolution right now."
        ),
        "Brand-Witty": (
            f"Okay, that's not the vibe we were going for. "
            f"{root_cause} Consider us officially on it — "
            "expect a plot twist soon."
        ),
    }


# ---------------------------------------------------------------------------
# CSV / JSON stream readers
# ---------------------------------------------------------------------------
def read_stream_source(file_path: str) -> List[Dict[str, object]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Stream source not found: {file_path}")
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    required = {"text", "timestamp", "followers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.to_dict(orient="records")


def simulate_live_stream(
    file_path: str, interval_seconds: int = 3
) -> Generator[Dict[str, object], None, None]:
    """Yield posts from CSV/JSON at *interval_seconds* to mimic a live stream."""
    posts = read_stream_source(file_path)
    for post in posts:
        yield post
        time.sleep(interval_seconds)
