"""Computes derived, stock-level metrics from raw price/news/sentiment data.

This module contains pure functions (no I/O) so it is easy to unit test:
it takes price history + sentiment scores in, and returns a DailyMetrics
object / DataFrame out.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.stock import DailyMetrics, SentimentScore, Stock
from utils.helpers import get_logger, safe_round

logger = get_logger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "at", "by", "as", "it", "its",
    "this", "that", "from", "after", "before", "over", "into", "up", "down",
    "new", "will", "has", "have", "had", "you", "your", "why", "how", "what",
    "stock", "stocks", "share", "shares", "market", "markets", "inc", "corp",
    "company", "co", "vs", "than", "more", "most", "top", "best", "says",
}


def compute_returns(price_history: pd.DataFrame) -> dict:
    """Compute 1/7/30/90-day percentage returns from a price history frame.

    ``price_history`` is expected to have a 'Date' and 'Close' column,
    sorted ascending by date (as returned by yahoo_service.fetch_price_history).
    """

    result = {"return_1d": None, "return_7d": None, "return_30d": None, "return_90d": None}
    if price_history is None or price_history.empty or "Close" not in price_history.columns:
        return result

    closes = price_history["Close"].dropna().reset_index(drop=True)
    if closes.empty:
        return result

    latest = closes.iloc[-1]

    for label, lookback in (("return_1d", 1), ("return_7d", 7), ("return_30d", 30), ("return_90d", 90)):
        idx = len(closes) - 1 - lookback
        if idx >= 0 and closes.iloc[idx] not in (0, None) and not pd.isna(closes.iloc[idx]):
            past = closes.iloc[idx]
            result[label] = safe_round(((latest - past) / past) * 100, 2)

    return result


def compute_sentiment_aggregates(sentiment_scores: List[SentimentScore]) -> dict:
    """Aggregate a list of per-article sentiment scores into stock-level stats.

    Sentiment Trend compares the average signed sentiment of the more-recent
    half of articles against the older half (by publish date).
    """

    result = {"avg_sentiment": None, "sentiment_trend": "Stable", "news_volume": 0}
    if not sentiment_scores:
        return result

    df = pd.DataFrame(
        [
            {"signed_score": s.signed_score, "published_at": s.published_at}
            for s in sentiment_scores
        ]
    )
    result["news_volume"] = len(df)
    result["avg_sentiment"] = safe_round(df["signed_score"].mean(), 3)

    dated = df.dropna(subset=["published_at"]).sort_values("published_at")
    if len(dated) >= 4:
        midpoint = len(dated) // 2
        older_avg = dated.iloc[:midpoint]["signed_score"].mean()
        recent_avg = dated.iloc[midpoint:]["signed_score"].mean()
        delta = recent_avg - older_avg
        if delta > 0.08:
            result["sentiment_trend"] = "Improving"
        elif delta < -0.08:
            result["sentiment_trend"] = "Declining"
        else:
            result["sentiment_trend"] = "Stable"
    return result


def compute_bullish_bearish_scores(returns: dict, avg_sentiment: Optional[float]) -> dict:
    """Blend momentum (returns) and sentiment into 0-100 bullish/bearish scores.

    Weighting: 30-day return is the primary momentum signal (short-term
    noise from 1-day is downweighted), and sentiment contributes up to half
    of the composite score.
    """

    momentum = 0.0
    weights_used = 0.0
    weight_map = {"return_1d": 0.10, "return_7d": 0.25, "return_30d": 0.40, "return_90d": 0.25}
    for key, weight in weight_map.items():
        value = returns.get(key)
        if value is not None:
            # Clip extreme moves so one outlier ticker doesn't dominate the scale.
            momentum += weight * float(np.clip(value, -50, 50))
            weights_used += weight
    momentum_norm = (momentum / weights_used) if weights_used else 0.0  # roughly -50..50

    sentiment = avg_sentiment if avg_sentiment is not None else 0.0  # -1..1

    # Composite in -100..100: momentum scaled to -50..50, sentiment scaled to -50..50
    composite = float(np.clip(momentum_norm, -50, 50)) + sentiment * 50
    composite = float(np.clip(composite, -100, 100))

    bullish_score = safe_round(max(composite, 0), 1)
    bearish_score = safe_round(max(-composite, 0), 1)
    return {"bullish_score": bullish_score, "bearish_score": bearish_score, "composite_score": composite}


def build_daily_metrics(
    stock: Stock,
    price_history: pd.DataFrame,
    sentiment_scores: List[SentimentScore],
) -> DailyMetrics:
    """Assemble a single DailyMetrics record for one ticker."""

    returns = compute_returns(price_history)
    sentiment_agg = compute_sentiment_aggregates(sentiment_scores)
    scores = compute_bullish_bearish_scores(returns, sentiment_agg["avg_sentiment"])

    return DailyMetrics(
        ticker=stock.ticker,
        current_price=stock.current_price,
        return_1d=returns["return_1d"],
        return_7d=returns["return_7d"],
        return_30d=returns["return_30d"],
        return_90d=returns["return_90d"],
        market_cap=stock.market_cap,
        avg_sentiment=sentiment_agg["avg_sentiment"],
        sentiment_trend=sentiment_agg["sentiment_trend"],
        news_volume=sentiment_agg["news_volume"],
        bullish_score=scores["bullish_score"],
        bearish_score=scores["bearish_score"],
        watchlist_rank=None,
        computed_at=datetime.now(timezone.utc),
    )


def rank_watchlist(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a 1-based Watchlist Rank ordered by (bullish - bearish) score desc."""

    if metrics_df.empty:
        return metrics_df

    df = metrics_df.copy()
    df["_composite"] = df["bullish_score"].fillna(0) - df["bearish_score"].fillna(0)
    df = df.sort_values("_composite", ascending=False).reset_index(drop=True)
    df["watchlist_rank"] = df.index + 1
    return df.drop(columns=["_composite"])


def extract_key_themes(headlines: List[str], top_n: int = 5) -> List[str]:
    """Extract the most frequent non-trivial words across a set of headlines.

    Purely local, frequency-based keyword extraction (no external LLM call) -
    used to surface "discussion themes" for the AI Insights page.
    """

    words: List[str] = []
    for headline in headlines:
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", headline or ""):
            token_l = token.lower()
            if token_l not in _STOPWORDS:
                words.append(token_l)

    if not words:
        return []

    counts = Counter(words)
    return [word.title() for word, _ in counts.most_common(top_n)]


def generate_ai_summary(
    ticker: str,
    company: str,
    metrics_row: pd.Series,
    news_subset: pd.DataFrame,
) -> Dict[str, object]:
    """Build a rule-based, investment-style summary for one stock.

    Entirely local logic (string templates driven by computed metrics) -
    no external LLM API call is made, per the AI Insights page spec.
    """

    avg_sentiment = metrics_row.get("avg_sentiment")
    trend = metrics_row.get("sentiment_trend") or "Stable"
    news_volume = int(metrics_row.get("news_volume") or 0)
    return_30d = metrics_row.get("return_30d")
    bullish_score = metrics_row.get("bullish_score") or 0
    bearish_score = metrics_row.get("bearish_score") or 0

    if avg_sentiment is None or pd.isna(avg_sentiment):
        sentiment_strength, sentiment_label = "neutral", "neutral"
    else:
        magnitude = abs(avg_sentiment)
        strength = "strongly" if magnitude > 0.4 else "moderately" if magnitude > 0.15 else "mildly"
        sentiment_label = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
        sentiment_strength = strength

    themes = extract_key_themes(news_subset["title"].tolist()) if not news_subset.empty else []

    overall_stance = "bullish" if bullish_score >= bearish_score else "bearish"

    return_phrase = (
        f"Price momentum is {'positive' if return_30d and return_30d > 0 else 'negative' if return_30d and return_30d < 0 else 'flat'} "
        f"with a 30-day return of {return_30d:+.2f}%." if return_30d is not None and not pd.isna(return_30d)
        else "Recent price momentum data is limited."
    )

    bullish_summary = (
        f"{company} currently exhibits {sentiment_strength} {sentiment_label} sentiment. "
        + (f"Key discussion themes: {', '.join(themes)}. " if themes else "")
        + f"Overall sentiment remains {overall_stance} based on {news_volume} recent news article(s) "
        f"with a {trend.lower()} trend. {return_phrase}"
    )

    positive_news = news_subset[news_subset.get("label") == "positive"] if not news_subset.empty else pd.DataFrame()
    negative_news = news_subset[news_subset.get("label") == "negative"] if not news_subset.empty else pd.DataFrame()

    key_opportunities = positive_news["title"].head(3).tolist() if not positive_news.empty else []
    key_concerns = negative_news["title"].head(3).tolist() if not negative_news.empty else []

    bearish_risks: List[str] = []
    if return_30d is not None and not pd.isna(return_30d) and return_30d < 0:
        bearish_risks.append(f"Price has declined {abs(return_30d):.2f}% over the last 30 days.")
    if trend == "Declining":
        bearish_risks.append("News sentiment trend is declining relative to prior coverage.")
    if bearish_score and bearish_score > bullish_score:
        bearish_risks.append("Composite bearish score currently exceeds the bullish score.")
    if not key_concerns and not bearish_risks:
        bearish_risks.append("No significant bearish signals detected in current data.")
    bearish_risks.extend(f"Negative coverage: “{t}”" for t in key_concerns)

    if not key_opportunities:
        key_opportunities = ["No strongly positive headlines identified in the current news window."]
    if not key_concerns:
        key_concerns = ["No strongly negative headlines identified in the current news window."]

    return {
        "ticker": ticker,
        "company": company,
        "sentiment_label": sentiment_label,
        "overall_stance": overall_stance,
        "themes": themes,
        "bullish_summary": bullish_summary,
        "bearish_risks": bearish_risks,
        "key_opportunities": key_opportunities,
        "key_concerns": key_concerns,
    }
