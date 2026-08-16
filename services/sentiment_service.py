"""AI/NLP sentiment analysis for news headlines using FinBERT.

The FinBERT model (ProsusAI/finbert) is loaded lazily and cached as a
Streamlit resource so it is only downloaded/initialized once per server
process. If transformers/torch are unavailable or the model cannot be
downloaded (e.g. no network access), the service transparently falls back
to a lightweight finance-lexicon sentiment scorer so the rest of the app
keeps working.
"""

from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from config import FINBERT_MODEL_NAME
from models.stock import NewsArticle, SentimentScore
from utils.helpers import get_logger

logger = get_logger(__name__)

_LABELS = ("positive", "negative", "neutral")

# Small finance-oriented lexicon used only as an offline fallback when
# FinBERT cannot be loaded (no internet access / transformers not installed).
_POSITIVE_WORDS = {
    "beat", "beats", "growth", "surge", "surges", "soar", "soars", "rally",
    "rallies", "record", "upgrade", "upgraded", "outperform", "bullish",
    "profit", "profits", "gain", "gains", "strong", "expansion", "buy",
    "positive", "optimistic", "innovation", "breakthrough", "boost",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "decline", "declines", "plunge", "plunges", "crash",
    "crashes", "downgrade", "downgraded", "underperform", "bearish", "loss",
    "losses", "weak", "lawsuit", "investigation", "recall", "sell", "sells",
    "negative", "pessimistic", "layoffs", "cut", "cuts", "fraud", "probe",
}


@st.cache_resource(show_spinner=False)
def _load_finbert_pipeline():
    """Load and cache the FinBERT sentiment-analysis pipeline.

    Returns None (instead of raising) if the model cannot be loaded, so
    callers can fall back gracefully.
    """

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        nlp = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
        )
        logger.info("FinBERT model '%s' loaded successfully", FINBERT_MODEL_NAME)
        return nlp
    except Exception as exc:  # noqa: BLE001 - broad: any load failure -> fallback
        logger.warning(
            "Could not load FinBERT model (%s). Falling back to lexicon-based "
            "sentiment analysis. Reason: %s", FINBERT_MODEL_NAME, exc,
        )
        return None


def _lexicon_sentiment(text: str) -> Tuple[str, float, float]:
    """Very small offline fallback: word-count based polarity."""

    words = [w.strip(".,!?;:()[]\"'").lower() for w in text.split()]
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)

    if pos == 0 and neg == 0:
        return "neutral", 0.55, 0.0

    total = pos + neg
    signed = (pos - neg) / total  # -1..1
    label = "positive" if signed > 0.1 else "negative" if signed < -0.1 else "neutral"
    confidence = min(0.5 + abs(signed) / 2, 0.95)
    return label, confidence, signed


def analyze_text(text: str) -> Tuple[str, float, float]:
    """Return (label, confidence_score, signed_score) for a single text."""

    text = (text or "").strip()
    if not text:
        return "neutral", 0.5, 0.0

    nlp = _load_finbert_pipeline()
    if nlp is None:
        return _lexicon_sentiment(text)

    try:
        raw = nlp(text[:512])
        # top_k=None returns a list of {label, score} for every class,
        # nested one level because we pass a single string.
        scores = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
        score_map = {item["label"].lower(): item["score"] for item in scores}

        label = max(score_map, key=score_map.get)
        confidence = score_map[label]
        signed = score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
        return label, float(confidence), float(signed)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FinBERT inference failed, using lexicon fallback: %s", exc)
        return _lexicon_sentiment(text)


def analyze_news_articles(articles: List[NewsArticle]) -> List[SentimentScore]:
    """Run sentiment analysis over a batch of news articles for one ticker."""

    results: List[SentimentScore] = []
    model_name = "finbert" if _load_finbert_pipeline() is not None else "lexicon-fallback"

    for article in articles:
        text = article.title if not article.summary else f"{article.title}. {article.summary}"
        label, score, signed = analyze_text(text)
        results.append(
            SentimentScore(
                ticker=article.ticker,
                title=article.title,
                label=label,
                score=score,
                signed_score=signed,
                published_at=article.published_at,
                model_used=model_name,
            )
        )
    return results
