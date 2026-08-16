"""Typed data models shared across the Stock Intelligence Dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Stock:
    """Core identity + fundamentals for a single ticker."""

    ticker: str
    company: str
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    analyst_target_price: Optional[float] = None
    recommendation: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    fetched_at: datetime = field(default_factory=_utc_now)
    fetch_error: Optional[str] = None


@dataclass
class NewsArticle:
    """A single news headline associated with a ticker."""

    ticker: str
    title: str
    publisher: Optional[str]
    link: Optional[str]
    published_at: Optional[datetime]
    summary: Optional[str] = None


@dataclass
class SentimentScore:
    """Sentiment analysis result for a single news article."""

    ticker: str
    title: str
    label: str  # positive | negative | neutral
    score: float  # confidence of the winning label, 0..1
    signed_score: float  # positive_prob - negative_prob, -1..1
    published_at: Optional[datetime]
    model_used: str = "finbert"


@dataclass
class DailyMetrics:
    """Aggregated, computed metrics for a stock at a point in time."""

    ticker: str
    current_price: Optional[float] = None
    return_1d: Optional[float] = None
    return_7d: Optional[float] = None
    return_30d: Optional[float] = None
    return_90d: Optional[float] = None
    market_cap: Optional[float] = None
    avg_sentiment: Optional[float] = None
    sentiment_trend: Optional[str] = None  # Improving | Declining | Stable
    news_volume: int = 0
    bullish_score: Optional[float] = None
    bearish_score: Optional[float] = None
    watchlist_rank: Optional[int] = None
    computed_at: datetime = field(default_factory=_utc_now)
