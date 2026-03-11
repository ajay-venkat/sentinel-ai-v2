"""Dynamic Sentinel AI Pipeline — fetches Reddit comments, streams data, watches for updates, and reports live metrics."""

import os
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from engine import RedditConfig, analyze_post, extract_keywords, fetch_reddit_posts, load_models
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
REDDIT_POLL = int(os.environ.get("SENTINEL_REDDIT_POLL", "60"))  # seconds between Reddit fetches

# Reddit credentials (set via env vars or prompted interactively)
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_SUBREDDIT = os.environ.get("REDDIT_SUBREDDIT", "technology")
REDDIT_LIMIT = int(os.environ.get("REDDIT_LIMIT", "25"))


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
# Fake Reddit comment generator (demo mode)
# ---------------------------------------------------------------------------
_FAKE_COMMENTS = [
    # Negative — Quality
    "This app crashes every single time I try to open settings. Completely unusable on Android.",
    "Latest update broke dark mode and now my eyes are bleeding at 2am. Thanks devs.",
    "Been getting data corruption errors since v4.3. Lost a week of work. Absolutely furious.",
    "Memory usage is insane — 4GB for a note-taking app? Is it running a blockchain in the background?",
    "The search feature returns completely irrelevant results. It's like it's searching a different database.",
    "App freezes for 30 seconds every time I try to export a PDF. This is enterprise software??",
    "Three major bugs in three weeks. QA team must be on permanent vacation.",
    "Sync has been broken for 5 days. Support says 'we're working on it.' Cool story.",
    "The new editor is so slow I can literally watch each character appear. Going back to Notepad.",
    "Just lost all my saved templates after the auto-update. No backup, no warning. Unacceptable.",
    # Negative — Price
    "$39/month for THIS? The free tier of the competitor does more. What a joke.",
    "They raised prices 40% and removed features. Classic bait and switch.",
    "The premium plan costs more than my Netflix, Spotify and gym combined. For a productivity app.",
    "Hidden fees everywhere. The advertised price is basically a lie.",
    "Just got charged $299 for annual renewal I never agreed to. Disputing with bank.",
    # Negative — Service
    "Support ticket open for 2 weeks. Zero response. Not even an automated 'we got your message'.",
    "Called support, got transferred 4 times, then disconnected. Peak customer service.",
    "The chatbot is useless — keeps telling me to 'check the FAQ' for a billing error.",
    "Waited 3 hours for live chat. Agent copy-pasted a generic response and closed the ticket.",
    "Tried to cancel subscription, got put through a 7-step retention maze. Dark patterns much?",
    # Positive — Quality
    "New update is genuinely great. Performance improvements are noticeable immediately.",
    "The real-time collaboration feature works flawlessly. Our team productivity doubled.",
    "Finally an app that handles large files without choking. Processed 2GB dataset in seconds.",
    "The new UI is clean, fast, and intuitive. Best update in years honestly.",
    "Offline mode actually works perfectly now. Used it on a 12-hour flight, zero issues.",
    "The API is incredibly well-designed. Integrated it with our pipeline in under an hour.",
    "Dark mode is perfect — proper OLED black, not that fake dark grey most apps use.",
    # Positive — Service
    "Support resolved my issue in 8 minutes. Actual human, knew exactly what to do. 10/10.",
    "The onboarding team walked us through enterprise setup personally. Incredible service.",
    "Got a full refund no questions asked within 24 hours. Respect.",
    "Their community forum is super active. Got help from a dev within an hour.",
    # Positive — Price
    "Free tier is incredibly generous. Gets 90% of what most people need.",
    "Switched from competitor and saving $200/year for better features. No brainer.",
    "The student discount makes this basically free. Huge respect for that.",
    # Neutral / Informational
    "Server maintenance tonight 11PM-3AM EST. Plan accordingly.",
    "New beta version available in the insider channel. Anyone tried it yet?",
    "PSA: the export bug workaround is to use CSV format instead of JSON.",
    "Changelog for v4.4 looks promising. 47 bug fixes and 3 new integrations.",
    "Anyone know if the Linux client supports Wayland yet?",
    # Sarcastic
    "Oh great another update that 'improves stability'. Can't wait to see what breaks this time.",
    "Love how 'premium support' means waiting 48 hours instead of 72. Worth every penny lol.",
    "Amazing that a billion-dollar company can't make a login page that works. Truly inspiring.",
    "Yeah the app is 'blazing fast' if you have a supercomputer and fiber internet.",
    "lol they call this an 'AI feature'? It literally just searches Google and summarizes badly.",
]


def generate_fake_reddit_batch(count: int = 10) -> list[dict]:
    """Generate a batch of realistic fake Reddit-style comments."""
    now = datetime.now(timezone.utc)
    batch = []
    selected = random.sample(_FAKE_COMMENTS, min(count, len(_FAKE_COMMENTS)))
    for i, text in enumerate(selected):
        batch.append({
            "text": text,
            "timestamp": (now - timedelta(seconds=random.randint(0, 300))).isoformat(),
            "followers": random.choice([100, 500, 1200, 3500, 8000, 15000, 25000, 50000]),
            "source": "reddit-demo",
        })
    return batch


def demo_live_loop(models, tracker: MetricsTracker) -> None:
    """Simulate a live Reddit stream using fake data — no credentials needed."""
    poll = max(REDDIT_POLL // 6, 5)  # faster for demo
    batch_size = 8

    print(f"\n{'=' * 60}")
    print(f"  🎭 DEMO MODE — Simulated Reddit Live Stream")
    print(f"  Generating {batch_size} fake comments every {poll}s")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'=' * 60}\n")

    used_comments: set[str] = set()
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n--- Demo cycle #{cycle} ({datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}) ---")

            # Generate comments not yet used
            available = [c for c in _FAKE_COMMENTS if c not in used_comments]
            if not available:
                print("  🔄 All demo comments used. Resetting pool…")
                used_comments.clear()
                available = _FAKE_COMMENTS[:]

            count = min(batch_size, len(available))
            posts = []
            selected = random.sample(available, count)
            now = datetime.now(timezone.utc)
            for text in selected:
                posts.append({
                    "text": text,
                    "timestamp": (now - timedelta(seconds=random.randint(0, 120))).isoformat(),
                    "followers": random.choice([200, 800, 2500, 6000, 12000, 22000, 40000]),
                    "source": "reddit-demo",
                })
                used_comments.add(text)

            print(f"  📥 Generated {len(posts)} simulated Reddit comments.")

            # Append to CSV
            saved = append_posts_to_csv(posts, DATA_SOURCE)
            if saved:
                print(f"  💾 Saved {saved} new rows to '{DATA_SOURCE}'.")

            # Analyse live
            print()
            stream_analyze(posts, models, tracker, start_idx=0)
            tracker.print_snapshot()

            if cycle % 2 == 0:
                tracker.print_keywords()
                tracker.print_ai_insights()

            print(f"\n  ⏳ Next batch in {poll}s…")
            time.sleep(poll)

    except KeyboardInterrupt:
        print("\n\n⏹  Demo mode stopped.")
        print("\n" + "=" * 60)
        print("  FINAL REPORT")
        print("=" * 60)
        tracker.print_snapshot()
        tracker.print_keywords()
        tracker.print_ai_insights()


# ---------------------------------------------------------------------------
# Reddit real-time fetcher
# ---------------------------------------------------------------------------
def get_reddit_config() -> RedditConfig:
    """Build Reddit config from env vars or interactive prompts."""
    client_id = REDDIT_CLIENT_ID
    client_secret = REDDIT_CLIENT_SECRET
    subreddit = REDDIT_SUBREDDIT

    if not client_id:
        client_id = input("  Reddit Client ID: ").strip()
    if not client_secret:
        client_secret = input("  Reddit Client Secret: ").strip()
    if not subreddit:
        subreddit = input("  Subreddit (default: technology): ").strip() or "technology"

    return RedditConfig(
        client_id=client_id,
        client_secret=client_secret,
        subreddit=subreddit,
        limit=REDDIT_LIMIT,
    )


def append_posts_to_csv(posts: list[dict], csv_path: str) -> int:
    """Append new Reddit posts to the CSV, skipping duplicates. Returns count of new rows."""
    path = Path(csv_path)
    if path.exists():
        existing_df = pd.read_csv(path)
        existing_texts = set(existing_df["text"].astype(str).tolist())
    else:
        existing_df = pd.DataFrame(columns=["timestamp", "text", "followers"])
        existing_texts = set()

    new_rows = []
    for post in posts:
        text = str(post.get("text", "")).strip()
        if text and text not in existing_texts:
            new_rows.append({
                "timestamp": post.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "text": text,
                "followers": post.get("followers", 1),
            })
            existing_texts.add(text)

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(path, index=False)
    return len(new_rows)


def fetch_and_analyze_reddit(models, tracker: MetricsTracker, cfg: RedditConfig) -> int:
    """Fetch Reddit comments, analyze them, append to CSV. Returns count of new posts."""
    print(f"\n🌐 Fetching up to {cfg.limit} comments from r/{cfg.subreddit}…")
    posts = fetch_reddit_posts(cfg)
    if not posts:
        print("  ⚠️  No Reddit data returned (check credentials or subreddit name).")
        return 0

    print(f"  📥 Fetched {len(posts)} comments from Reddit.")

    # Append to CSV
    new_count = append_posts_to_csv(posts, DATA_SOURCE)
    if new_count > 0:
        print(f"  💾 Appended {new_count} new unique posts to '{DATA_SOURCE}'.")
    else:
        print("  ℹ️  No new unique posts (all duplicates).")

    # Analyze new posts
    new_posts = [p for p in posts if str(p.get("text", "")).strip()]
    if new_posts:
        print(f"\n  📊 Analysing {len(new_posts)} Reddit comments…\n")
        stream_analyze(new_posts, models, tracker, start_idx=0)
        tracker.print_snapshot()
        tracker.print_keywords()
        tracker.print_ai_insights()

    return new_count


def reddit_live_loop(models, tracker: MetricsTracker, cfg: RedditConfig) -> None:
    """Continuously fetch Reddit comments at intervals and analyze them in real-time."""
    print(f"\n{'=' * 60}")
    print(f"  🔴 LIVE REDDIT MODE — r/{cfg.subreddit}")
    print(f"  Polling every {REDDIT_POLL}s | Limit: {cfg.limit} comments/fetch")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'=' * 60}\n")

    seen_texts: set[str] = set()
    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n--- Fetch cycle #{cycle} ({datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}) ---")

            posts = fetch_reddit_posts(cfg)
            if not posts:
                print("  ⚠️  No data returned. Retrying next cycle…")
                time.sleep(REDDIT_POLL)
                continue

            # Filter to truly new posts
            new_posts = []
            for p in posts:
                text = str(p.get("text", "")).strip()
                if text and text not in seen_texts:
                    new_posts.append(p)
                    seen_texts.add(text)

            if not new_posts:
                print(f"  ℹ️  {len(posts)} fetched, 0 new. Waiting…")
                time.sleep(REDDIT_POLL)
                continue

            print(f"  📥 {len(new_posts)} new comments (out of {len(posts)} fetched).")

            # Save to CSV
            saved = append_posts_to_csv(new_posts, DATA_SOURCE)
            if saved:
                print(f"  💾 Saved {saved} new rows to '{DATA_SOURCE}'.")

            # Analyze live
            print()
            stream_analyze(new_posts, models, tracker, start_idx=0)
            tracker.print_snapshot()

            # Show keywords every 3 cycles
            if cycle % 3 == 0:
                tracker.print_keywords()
                tracker.print_ai_insights()

            print(f"\n  ⏳ Next fetch in {REDDIT_POLL}s…")
            time.sleep(REDDIT_POLL)

    except KeyboardInterrupt:
        print("\n\n⏹  Reddit live mode stopped.")
        tracker.print_snapshot()
        tracker.print_keywords()
        tracker.print_ai_insights()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  🛡️  SENTINEL AI — Dynamic Analysis Pipeline")
    print("=" * 60)
    print()
    print("  Modes:")
    print("    1. Analyse CSV + watch for updates")
    print("    2. Fetch Reddit comments (one-time) + analyse")
    print("    3. Live Reddit stream (continuous polling)")
    print("    4. Full pipeline: CSV + Reddit live")
    print("    5. 🎭 DEMO — Simulated Reddit live (no credentials needed)")
    print()

    mode = input("  Select mode [1/2/3/4/5] (default=5): ").strip() or "5"

    print(f"\n  Data source:  {DATA_SOURCE}")
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
        f"aspect={'HF' if models.aspect_pipe else 'fallback'})\n"
    )

    tracker = MetricsTracker()

    # --- MODE 1: CSV analysis + file watcher ---
    if mode == "1":
        run_csv_analysis(models, tracker)
        watch_and_update(models, tracker)

    # --- MODE 2: One-time Reddit fetch + analyse ---
    elif mode == "2":
        print("🌐 Reddit Configuration:")
        cfg = get_reddit_config()
        # First analyse existing CSV if present
        try:
            posts = read_stream_source(DATA_SOURCE)
            print(f"\n📊 Analysing {len(posts)} existing posts from CSV…\n")
            stream_analyze(posts, models, tracker)
        except (FileNotFoundError, ValueError):
            print("  No existing CSV data. Starting fresh.\n")
        # Fetch and analyse Reddit
        fetch_and_analyze_reddit(models, tracker, cfg)
        tracker.print_snapshot()
        tracker.print_keywords()
        tracker.print_ai_insights()

    # --- MODE 3: Continuous Reddit live stream ---
    elif mode == "3":
        print("🌐 Reddit Configuration:")
        cfg = get_reddit_config()
        reddit_live_loop(models, tracker, cfg)

    # --- MODE 4: CSV + Reddit live (full pipeline) ---
    elif mode == "4":
        print("🌐 Reddit Configuration:")
        cfg = get_reddit_config()
        # Analyse existing CSV first
        run_csv_analysis(models, tracker)
        # Then enter Reddit live loop
        reddit_live_loop(models, tracker, cfg)

    # --- MODE 5: Demo mode (fake Reddit data) ---
    elif mode == "5":
        # Optionally analyse existing CSV first
        try:
            posts = read_stream_source(DATA_SOURCE)
            print(f"📊 Analysing {len(posts)} existing posts from CSV…\n")
            stream_analyze(posts, models, tracker)
            tracker.print_snapshot()
        except (FileNotFoundError, ValueError):
            print("  No existing CSV data. Starting with demo data only.\n")
        # Enter demo live loop
        demo_live_loop(models, tracker)

    else:
        print("❌ Invalid mode. Use 1, 2, 3, 4, or 5.")
        sys.exit(1)


def run_csv_analysis(models, tracker: MetricsTracker) -> None:
    """Analyse all posts from the CSV data source."""
    print(f"📊 Streaming analysis of '{DATA_SOURCE}'…\n")
    try:
        posts = read_stream_source(DATA_SOURCE)
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_SOURCE}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Data format error: {e}")
        sys.exit(1)

    stream_analyze(posts, models, tracker)
    tracker.print_snapshot()
    tracker.print_keywords()
    tracker.print_ai_insights()


if __name__ == "__main__":
    main()
