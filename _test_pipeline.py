"""Dynamic Sentinel AI Pipeline — streams data, watches for updates, and reports live metrics."""

import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd

from engine import analyze_post, extract_keywords, load_models
from utils import (
    generate_ai_response,
    read_stream_source,
    summarize_root_cause,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_SOURCE = os.environ.get("SENTINEL_DATA", str(Path("data") / "sample_posts.csv"))
POLL_INTERVAL = int(os.environ.get("SENTINEL_POLL", "5"))  # seconds between file-change checks
STREAM_DELAY = float(os.environ.get("SENTINEL_DELAY", "0.3"))  # delay per post during streaming


# ---------------------------------------------------------------------------
# Rolling metrics tracker
# ---------------------------------------------------------------------------
class MetricsTracker:
    """Maintains live-updated aggregate metrics as posts are processed."""

    def __init__(self):
        self.rows: list[dict] = []
        self.sentiment_counts: Counter = Counter()
        self.aspect_counts: Counter = Counter()
        self.sarcasm_count: int = 0
        self.crisis_sum: float = 0.0
        self.crisis_max: float = 0.0
        self.high_crisis_posts: list[dict] = []

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def avg_crisis(self) -> float:
        return self.crisis_sum / self.total if self.total else 0.0

    def ingest(self, result: dict) -> None:
        """Add one analysed post and update all rolling counters."""
        self.rows.append(result)
        self.sentiment_counts[result["final_sentiment"]] += 1
        self.aspect_counts[result["aspect"]] += 1
        if result["is_sarcastic"]:
            self.sarcasm_count += 1
        score = result["crisis_score"]
        self.crisis_sum += score
        if score > self.crisis_max:
            self.crisis_max = score
        if score > 70:
            self.high_crisis_posts.append(result)

    def net_sentiment_pct(self) -> float:
        pos = self.sentiment_counts.get("Positive", 0)
        neg = self.sentiment_counts.get("Negative", 0)
        return round((pos - neg) / max(self.total, 1) * 100, 2)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows) if self.rows else pd.DataFrame()

    def print_snapshot(self) -> None:
        """Print the current state of all tracked metrics."""
        print(f"\n{'=' * 60}")
        print(f"  LIVE METRICS  |  Posts analysed: {self.total}")
        print(f"{'=' * 60}")
        print(f"  Net Sentiment:   {self.net_sentiment_pct():+.1f}%")
        print(f"  Avg Crisis:      {self.avg_crisis:.2f}")
        print(f"  Max Crisis:      {self.crisis_max:.2f}")
        print(f"  High-Crisis (>70): {len(self.high_crisis_posts)}")
        print(f"  Sarcasm Detected:  {self.sarcasm_count}")
        print(f"  Sentiment Dist:  ", end="")
        for label in ("Positive", "Negative", "Informational"):
            print(f"{label}={self.sentiment_counts.get(label, 0)}  ", end="")
        print()
        print(f"  Aspect Dist:     ", end="")
        for label, cnt in self.aspect_counts.most_common():
            print(f"{label}={cnt}  ", end="")
        print()

    def print_keywords(self, top_n: int = 8) -> None:
        df = self.as_dataframe()
        if df.empty:
            return
        kws = extract_keywords(df, top_n=top_n)
        if kws:
            print(f"\n  Top Keywords:")
            for kw in kws:
                bar = "█" * min(kw["count"], 30)
                print(f"    {kw['keyword']:15s}  {bar} {kw['count']}  ({kw['sentiment']})")

    def print_ai_insights(self) -> None:
        negatives = [r["raw_text"] for r in self.rows if r["final_sentiment"] == "Negative"]
        if negatives:
            root = summarize_root_cause(negatives)
            print(f"\n  Root Cause:  {root}")
        if self.high_crisis_posts:
            worst = max(self.high_crisis_posts, key=lambda r: r["crisis_score"])
            root = summarize_root_cause(negatives) if negatives else ""
            responses = generate_ai_response(worst["raw_text"], root)
            print(f"\n  AI Response for worst crisis post (score={worst['crisis_score']}):")
            for tone, text in responses.items():
                print(f"    [{tone}] {text[:120]}...")


# ---------------------------------------------------------------------------
# Streaming processor
# ---------------------------------------------------------------------------
def stream_analyze(posts: list[dict], models, tracker: MetricsTracker, start_idx: int = 0) -> bool:
    """Analyse posts one-by-one from start_idx, updating metrics live.

    Returns True if all posts were processed, False if interrupted.
    """
    total = len(posts)
    try:
        for i in range(start_idx, total):
            post = posts[i]
            try:
                result = analyze_post(post, models)
                tracker.ingest(result)

                # Per-post live line
                tag = result["sentiment_tag"]
                emoji = {"Positive": "🟢", "Negative": "🔴", "Sarcastic/Critical": "🕵️"}.get(tag, "⚪")
                crisis_bar = "!" * min(int(result["crisis_score"] / 10), 10)
                print(
                    f"  {emoji} [{i + 1}/{total}] "
                    f"sentiment={result['final_sentiment']:15s} "
                    f"aspect={result['aspect']:8s} "
                    f"crisis={result['crisis_score']:5.1f} [{crisis_bar:10s}]"
                    f"{'  🕵️ SARCASM' if result['is_sarcastic'] else ''}"
                )
            except Exception as e:
                print(f"  ❌ Post {i + 1} ERROR: {e}")
                traceback.print_exc()

            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)

            # Periodic snapshot every 10 posts
            if (i + 1) % 10 == 0:
                tracker.print_snapshot()
    except KeyboardInterrupt:
        print(f"\n⏹  Streaming interrupted at post {i + 1}/{total}.")
        return False
    return True


# ---------------------------------------------------------------------------
# File watcher — detects new rows appended to the CSV/JSON
# ---------------------------------------------------------------------------
def get_file_state(path: str) -> tuple[float, int]:
    """Return (mtime, row_count) for change detection."""
    p = Path(path)
    if not p.exists():
        return (0.0, 0)
    mtime = p.stat().st_mtime
    try:
        posts = read_stream_source(path)
        return (mtime, len(posts))
    except Exception:
        return (mtime, 0)


def watch_and_update(models, tracker: MetricsTracker) -> None:
    """Poll the data file and process any newly appended rows."""
    print(f"\n👁️  Watching '{DATA_SOURCE}' for changes (poll every {POLL_INTERVAL}s)…")
    print("   Press Ctrl+C to stop.\n")

    prev_mtime, prev_count = get_file_state(DATA_SOURCE)

    try:
        while True:
            time.sleep(POLL_INTERVAL)
            cur_mtime, cur_count = get_file_state(DATA_SOURCE)

            if cur_mtime == prev_mtime:
                continue  # no file change

            if cur_count > prev_count:
                new_rows = cur_count - prev_count
                print(f"\n📥  Detected {new_rows} new row(s) in data file!")
                posts = read_stream_source(DATA_SOURCE)
                stream_analyze(posts, models, tracker, start_idx=prev_count)
                tracker.print_snapshot()
                tracker.print_keywords()
                tracker.print_ai_insights()
                prev_count = cur_count
            elif cur_count < prev_count:
                # File was rewritten — full re-analysis
                print(f"\n🔄  Data file rewritten ({prev_count} → {cur_count} rows). Re-analysing…")
                tracker.__init__()
                posts = read_stream_source(DATA_SOURCE)
                stream_analyze(posts, models, tracker)
                tracker.print_snapshot()
                tracker.print_keywords()
                tracker.print_ai_insights()
                prev_count = cur_count

            prev_mtime = cur_mtime

    except KeyboardInterrupt:
        print("\n\n⏹  Watcher stopped.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  🛡️  SENTINEL AI — Dynamic Analysis Pipeline")
    print("=" * 60)
    print(f"  Data source:  {DATA_SOURCE}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print()

    # --- Load models ---
    print("⏳ Loading models…")
    t0 = time.time()
    models = load_models()
    t1 = time.time()
    print(
        f"✅ Models loaded in {t1 - t0:.1f}s  "
        f"(sentiment={'HF' if models.sentiment_pipe else 'fallback'}, "
        f"aspect={'HF' if models.aspect_pipe else 'fallback'})"
    )

    # --- Initial analysis: stream through all existing data ---
    print(f"\n📊 Streaming analysis of '{DATA_SOURCE}'…\n")
    try:
        posts = read_stream_source(DATA_SOURCE)
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_SOURCE}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Data format error: {e}")
        sys.exit(1)

    tracker = MetricsTracker()
    stream_analyze(posts, models, tracker)

    # --- Full report ---
    tracker.print_snapshot()
    tracker.print_keywords()
    tracker.print_ai_insights()

    # --- Enter watch mode for dynamic updates ---
    watch_and_update(models, tracker)


if __name__ == "__main__":
    main()
