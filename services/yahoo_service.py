"""Yahoo Finance data access layer, built on top of yfinance.

All network calls are wrapped in try/except so that a single failed ticker
(bad symbol, delisted stock, network hiccup) never crashes the whole batch.
Results are cached with Streamlit's cache_data to avoid hammering Yahoo
Finance on every rerun.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from config import (
    INFO_CACHE_TTL_SECONDS,
    MAX_NEWS_PER_TICKER,
    NEWS_CACHE_TTL_SECONDS,
    PRICE_CACHE_TTL_SECONDS,
    PRICE_HISTORY_PERIOD,
)
from models.stock import NewsArticle, Stock
from utils.helpers import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=INFO_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_stock_info(ticker: str, company_fallback: str = "") -> Stock:
    """Fetch current fundamentals for a single ticker.

    Returns a Stock object even on failure; fetch_error is populated so
    callers can surface a friendly message instead of crashing.
    """

    ticker = ticker.strip().upper()
    try:
        yt = yf.Ticker(ticker)
        info = yt.info or {}

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # yfinance sometimes returns a near-empty dict for invalid tickers
            fast = getattr(yt, "fast_info", None)
            if not fast or fast.get("lastPrice") is None:
                raise ValueError(f"No market data returned for '{ticker}'")

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or _safe_fast_info(yt, "lastPrice")
        )

        return Stock(
            ticker=ticker,
            company=info.get("longName") or info.get("shortName") or company_fallback or ticker,
            current_price=current_price,
            previous_close=info.get("previousClose") or info.get("regularMarketPreviousClose"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE") or info.get("forwardPE"),
            week52_high=info.get("fiftyTwoWeekHigh"),
            week52_low=info.get("fiftyTwoWeekLow"),
            analyst_target_price=info.get("targetMeanPrice"),
            recommendation=_normalize_recommendation(info.get("recommendationKey")),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency=info.get("currency", "USD"),
            fetched_at=datetime.now(timezone.utc),
        )
    except Exception as exc:  # noqa: BLE001 - yfinance raises many undocumented types
        logger.warning("Failed to fetch info for %s: %s", ticker, exc)
        return Stock(
            ticker=ticker,
            company=company_fallback or ticker,
            fetched_at=datetime.now(timezone.utc),
            fetch_error=str(exc),
        )


def _safe_fast_info(yt: "yf.Ticker", key: str) -> Optional[float]:
    try:
        fast = yt.fast_info
        return fast.get(key)
    except Exception:  # noqa: BLE001
        return None


def _normalize_recommendation(key: Optional[str]) -> str:
    if not key:
        return "N/A"
    mapping = {
        "strong_buy": "Strong Buy",
        "buy": "Buy",
        "hold": "Hold",
        "sell": "Sell",
        "strong_sell": "Strong Sell",
        "none": "N/A",
    }
    return mapping.get(key.lower(), key.replace("_", " ").title())


@st.cache_data(ttl=PRICE_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_price_history(ticker: str, period: str = PRICE_HISTORY_PERIOD) -> pd.DataFrame:
    """Fetch historical OHLCV data. Returns an empty DataFrame on failure."""

    ticker = ticker.strip().upper()
    try:
        yt = yf.Ticker(ticker)
        hist = yt.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            logger.warning("Empty price history for %s", ticker)
            return pd.DataFrame()
        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        return hist
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch price history for %s: %s", ticker, exc)
        return pd.DataFrame()


@st.cache_data(ttl=NEWS_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_news(ticker: str, limit: int = MAX_NEWS_PER_TICKER) -> List[NewsArticle]:
    """Fetch recent news headlines for a ticker. Returns [] on failure."""

    ticker = ticker.strip().upper()
    try:
        yt = yf.Ticker(ticker)
        raw_news = yt.news or []
        articles: List[NewsArticle] = []
        for item in raw_news[:limit]:
            content = item.get("content", item)  # yfinance schema varies by version
            title = content.get("title") or item.get("title")
            if not title:
                continue
            publisher = (
                (content.get("provider") or {}).get("displayName")
                if isinstance(content.get("provider"), dict)
                else item.get("publisher")
            )
            link = (
                (content.get("canonicalUrl") or {}).get("url")
                if isinstance(content.get("canonicalUrl"), dict)
                else item.get("link")
            )
            pub_date = content.get("pubDate") or item.get("providerPublishTime")
            published_at = _parse_published(pub_date)
            summary = content.get("summary") or content.get("description")

            articles.append(
                NewsArticle(
                    ticker=ticker,
                    title=title,
                    publisher=publisher,
                    link=link,
                    published_at=published_at,
                    summary=summary,
                )
            )
        return articles
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch news for %s: %s", ticker, exc)
        return []


def _parse_published(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:  # noqa: BLE001
        return None
