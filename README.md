# Sentinel AI

Aspect-based sentiment and crisis prediction dashboard built with Streamlit.

## Features
- Real-time stream simulation from CSV/JSON (`simulate_live_stream`) at 3-second intervals.
- Sentiment using `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- Aspect detection using `facebook/bart-large-mnli`.
- Sarcasm correction layer for positive sentiment with negative context words.
- Crisis risk score:
  - `R = (Sentiment_Intensity * Reach_Factor) * Velocity`
- Root-cause summary from last 50 negative mentions.
- Auto-generated responses in 3 tones: Professional, Empathic, Brand-Witty.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Python Version
- Recommended: Python 3.10 to 3.12 for full transformer backend support.
- Python 3.13 can run the app, but `torch` may be unavailable and fallback NLP logic will be used.

## Data Format
Input CSV/JSON should contain:
- `timestamp` (ISO format preferred)
- `text`
- `reach`

## Notes
- Empty strings and emoji-only posts are handled safely.
- If model/API calls fail, the app uses fallback rules and keeps running.
- On Python versions without available `torch` wheels, transformer backends may be skipped and heuristic fallback logic will be used automatically.
