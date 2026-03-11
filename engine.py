"""engine.py – Model loading, sentiment/aspect analysis, crisis scoring, and Reddit integration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils import detect_sarcasm, preprocess_text

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful fallback when torch is unavailable)
# ---------------------------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

try:
    import praw
except Exception:
    praw = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ASPECT_MODEL = "facebook/bart-large-mnli"
ASPECT_LABELS = ["Price", "Quality", "Service"]


# ---------------------------------------------------------------------------
# Keyword-based fallback helpers
# ---------------------------------------------------------------------------
_NEG_WORDS = ["bad", "worst", "broken", "fail", "garbage", "hate", "awful", "terrible", "slow", "useless"]
_POS_WORDS = ["great", "love", "excellent", "awesome", "best", "happy", "fast", "good", "amazing", "reliable"]

_ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Price": ["price", "cost", "expensive", "cheap", "value", "pricing", "afford", "subscription", "fee"],
    "Quality": ["quality", "build", "defect", "broken", "reliable", "buggy", "works", "crash", "bug"],
    "Service": ["support", "help", "service", "agent", "ticket", "response", "customer", "refund"],
}


def _fallback_sentiment(text: str) -> Dict[str, Union[float, str]]:
    lower = text.lower()
    neg = sum(1 for w in _NEG_WORDS if w in lower)
    pos = sum(1 for w in _POS_WORDS if w in lower)
    if neg > pos:
        return {"label": "negative", "score": min(0.95, 0.55 + 0.1 * neg)}
    if pos > neg:
        return {"label": "positive", "score": min(0.95, 0.55 + 0.1 * pos)}
    return {"label": "neutral", "score": 0.6}


def _fallback_aspect(text: str) -> Dict[str, Any]:
    lower = text.lower()
    best_label, best_hits = "Quality", 0
    for label, words in _ASPECT_KEYWORDS.items():
        hits = sum(1 for w in words if w in lower)
        if hits > best_hits:
            best_label, best_hits = label, hits
    return {"labels": [best_label], "scores": [0.7]}


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    sentiment_pipe: Any = None
    aspect_pipe: Any = None

    def predict_sentiment(self, text: str) -> Dict[str, Union[float, str]]:
        if not text.strip():
            return {"label": "neutral", "score": 1.0}
        if self.sentiment_pipe is None:
            return _fallback_sentiment(text)
        try:
            result = self.sentiment_pipe(text)[0]
            return {"label": str(result["label"]).lower(), "score": float(result["score"])}
        except Exception:
            return _fallback_sentiment(text)

    def predict_aspect(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"labels": ["Quality"], "scores": [1.0]}
        if self.aspect_pipe is None:
            return _fallback_aspect(text)
        try:
            return self.aspect_pipe(sequences=text, candidate_labels=ASPECT_LABELS, multi_label=False)
        except Exception:
            return _fallback_aspect(text)


@lru_cache(maxsize=1)
def load_models() -> ModelBundle:
    if hf_pipeline is None:
        return ModelBundle()
    try:
        s_pipe = hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    except Exception:
        s_pipe = None
    try:
        a_pipe = hf_pipeline("zero-shot-classification", model=ASPECT_MODEL)
    except Exception:
        a_pipe = None
    return ModelBundle(sentiment_pipe=s_pipe, aspect_pipe=a_pipe)


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    low = label.lower()
    if "neg" in low:
        return "Negative"
    if "pos" in low:
        return "Positive"
    return "Neutral"


# ---------------------------------------------------------------------------
# Crisis Velocity Score
# ---------------------------------------------------------------------------
def crisis_velocity(sentiment_score: float, final_label: str, followers: float) -> float:
    """Crisis Velocity = Negative Sentiment Intensity × Follower Count.

    Scaled to 0-100 via a sigmoid so the gauge stays interpretable.
    Only negative / sarcastic posts contribute meaningfully.
    """
    if final_label not in ("Negative", "Sarcastic/Critical"):
        raw = sentiment_score * 0.05 * max(followers, 1)
    else:
        raw = sentiment_score * max(followers, 1)
    sigmoid = 1.0 / (1.0 + np.exp(-0.0008 * (raw - 3000)))
    return round(float(sigmoid * 100), 2)


# ---------------------------------------------------------------------------
# Full post analysis
# ---------------------------------------------------------------------------
def analyze_post(post: Dict[str, Any], models: ModelBundle) -> Dict[str, Any]:
    raw_text = str(post.get("text", "") or "")
    clean_text = preprocess_text(raw_text)

    sent_out = models.predict_sentiment(clean_text)
    base_label = normalize_label(str(sent_out.get("label", "neutral")))
    base_score = float(sent_out.get("score", 0.5))

    sarcastic = detect_sarcasm(raw_text, base_label)
    if sarcastic:
        final_sentiment = "Negative"
        sentiment_tag = "Sarcastic/Critical"
    elif base_label == "Neutral":
        final_sentiment = "Informational"
        sentiment_tag = "Informational"
    else:
        final_sentiment = base_label
        sentiment_tag = base_label

    aspect_out = models.predict_aspect(clean_text)
    aspect = str(aspect_out.get("labels", ["Quality"])[0])

    now_ts = pd.to_datetime(post.get("timestamp", datetime.now(timezone.utc)))
    if pd.isna(now_ts):
        now_ts = pd.Timestamp.now(tz="UTC")

    followers = float(post.get("followers", 1) or 1)
    crisis = crisis_velocity(base_score, final_sentiment, followers)

    return {
        "timestamp": now_ts,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "aspect": aspect,
        "base_sentiment": base_label,
        "base_score": round(base_score, 4),
        "sentiment_tag": sentiment_tag,
        "final_sentiment": final_sentiment,
        "is_sarcastic": sarcastic,
        "followers": followers,
        "crisis_score": crisis,
        "source": str(post.get("source", "csv")),
    }


# ---------------------------------------------------------------------------
# Reddit (PRAW) integration
# ---------------------------------------------------------------------------
@dataclass
class RedditConfig:
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "SentinelAI/1.0"
    subreddit: str = "technology"
    limit: int = 25


def fetch_reddit_posts(cfg: RedditConfig) -> List[Dict[str, Any]]:
    """Pull recent comments from r/technology (or configured subreddit).

    Returns an empty list silently when PRAW is unavailable or credentials
    are missing, so the dashboard always stays functional.
    """
    if praw is None:
        return []
    if not cfg.client_id or not cfg.client_secret:
        return []

    try:
        reddit = praw.Reddit(
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
            user_agent=cfg.user_agent,
        )
        subreddit = reddit.subreddit(cfg.subreddit)
        posts: List[Dict[str, Any]] = []
        for comment in subreddit.comments(limit=cfg.limit):
            posts.append({
                "text": str(comment.body or ""),
                "timestamp": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                "followers": 1,  # Reddit doesn't expose per-user follower counts on comments
                "source": "reddit",
            })
        return posts
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Keyword extraction (for Top-Keywords brick)
# ---------------------------------------------------------------------------
def extract_keywords(df: pd.DataFrame, top_n: int = 8) -> List[Dict[str, Any]]:
    """Return top-N keywords with their dominant sentiment polarity and count."""
    from utils import STOPWORDS, preprocess_text as _prep
    from collections import Counter

    word_sentiments: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        words = [
            w for w in re.findall(r"[a-z']+", _prep(str(row.get("raw_text", ""))).lower())
            if w not in STOPWORDS and len(w) > 2
        ]
        sentiment = str(row.get("final_sentiment", "Neutral"))
        for w in set(words):
            word_sentiments.setdefault(w, []).append(sentiment)

    results = []
    for word, sents in sorted(word_sentiments.items(), key=lambda x: -len(x[1])):
        if len(results) >= top_n:
            break
        total = len(sents)
        neg = sents.count("Negative")
        pos = sents.count("Positive")
        dominant = "Negative" if neg >= pos else "Positive" if pos > neg else "Neutral"
        results.append({"keyword": word, "count": total, "sentiment": dominant,
                         "neg_ratio": round(neg / total, 2) if total else 0})
    return results