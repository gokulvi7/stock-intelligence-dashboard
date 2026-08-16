"""SQLite persistence layer.

Creates the schema automatically on first use and exposes small, explicit
functions for reading/writing stocks, news articles, sentiment scores, and
daily metrics. Uses plain sqlite3 (no ORM) to keep the dependency footprint
small and behaviour easy to reason about.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional

import pandas as pd

from config import DB_PATH
from models.stock import DailyMetrics, NewsArticle, SentimentScore, Stock
from utils.helpers import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    company TEXT,
    current_price REAL,
    previous_close REAL,
    market_cap REAL,
    pe_ratio REAL,
    week52_high REAL,
    week52_low REAL,
    analyst_target_price REAL,
    recommendation TEXT,
    sector TEXT,
    industry TEXT,
    currency TEXT,
    fetch_error TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT,
    link TEXT,
    published_at TEXT,
    summary TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, title, published_at)
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    signed_score REAL NOT NULL,
    published_at TEXT,
    model_used TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, title)
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    current_price REAL,
    return_1d REAL,
    return_7d REAL,
    return_30d REAL,
    return_90d REAL,
    market_cap REAL,
    avg_sentiment REAL,
    sentiment_trend TEXT,
    news_volume INTEGER,
    bullish_score REAL,
    bearish_score REAL,
    watchlist_rank INTEGER,
    computed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_metrics_ticker ON daily_metrics(ticker);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, ensuring the schema exists."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables/indices if they do not already exist."""

    try:
        with get_connection() as conn:
            conn.executescript(SCHEMA)
        logger.info("Database schema ready at %s", DB_PATH)
    except sqlite3.Error:
        logger.exception("Failed to initialize database schema")
        raise


def upsert_stock(stock: Stock) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO stocks (
                    ticker, company, current_price, previous_close, market_cap,
                    pe_ratio, week52_high, week52_low, analyst_target_price,
                    recommendation, sector, industry, currency, fetch_error, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    company=excluded.company,
                    current_price=excluded.current_price,
                    previous_close=excluded.previous_close,
                    market_cap=excluded.market_cap,
                    pe_ratio=excluded.pe_ratio,
                    week52_high=excluded.week52_high,
                    week52_low=excluded.week52_low,
                    analyst_target_price=excluded.analyst_target_price,
                    recommendation=excluded.recommendation,
                    sector=excluded.sector,
                    industry=excluded.industry,
                    currency=excluded.currency,
                    fetch_error=excluded.fetch_error,
                    fetched_at=excluded.fetched_at
                """,
                (
                    stock.ticker, stock.company, stock.current_price, stock.previous_close,
                    stock.market_cap, stock.pe_ratio, stock.week52_high, stock.week52_low,
                    stock.analyst_target_price, stock.recommendation, stock.sector,
                    stock.industry, stock.currency, stock.fetch_error,
                    stock.fetched_at.isoformat(),
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to upsert stock %s", stock.ticker)


def insert_news_articles(articles: Iterable[NewsArticle]) -> None:
    articles = list(articles)
    if not articles:
        return
    try:
        with get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO news_articles
                    (ticker, title, publisher, link, published_at, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        a.ticker, a.title, a.publisher, a.link,
                        a.published_at.isoformat() if a.published_at else None,
                        a.summary,
                    )
                    for a in articles
                ],
            )
    except sqlite3.Error:
        logger.exception("Failed to insert news articles")


def insert_sentiment_scores(scores: Iterable[SentimentScore]) -> None:
    scores = list(scores)
    if not scores:
        return
    try:
        with get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO sentiment_scores
                    (ticker, title, label, score, signed_score, published_at, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.ticker, s.title, s.label, s.score, s.signed_score,
                        s.published_at.isoformat() if s.published_at else None,
                        s.model_used,
                    )
                    for s in scores
                ],
            )
    except sqlite3.Error:
        logger.exception("Failed to insert sentiment scores")


def insert_daily_metrics(metrics: DailyMetrics) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_metrics (
                    ticker, current_price, return_1d, return_7d, return_30d, return_90d,
                    market_cap, avg_sentiment, sentiment_trend, news_volume,
                    bullish_score, bearish_score, watchlist_rank, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.ticker, metrics.current_price, metrics.return_1d,
                    metrics.return_7d, metrics.return_30d, metrics.return_90d,
                    metrics.market_cap, metrics.avg_sentiment, metrics.sentiment_trend,
                    metrics.news_volume, metrics.bullish_score, metrics.bearish_score,
                    metrics.watchlist_rank, metrics.computed_at.isoformat(),
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to insert daily metrics for %s", metrics.ticker)


def fetch_news_for_ticker(ticker: str, limit: int = 50) -> pd.DataFrame:
    try:
        with get_connection() as conn:
            return pd.read_sql_query(
                """
                SELECT n.ticker, n.title, n.publisher, n.link, n.published_at, n.summary,
                       s.label, s.score, s.signed_score
                FROM news_articles n
                LEFT JOIN sentiment_scores s
                    ON n.ticker = s.ticker AND n.title = s.title
                WHERE n.ticker = ?
                ORDER BY n.published_at DESC
                LIMIT ?
                """,
                conn,
                params=(ticker, limit),
            )
    except sqlite3.Error:
        logger.exception("Failed to fetch news for %s", ticker)
        return pd.DataFrame()


def fetch_latest_metrics(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        with get_connection() as conn:
            query = """
                SELECT dm.*
                FROM daily_metrics dm
                INNER JOIN (
                    SELECT ticker, MAX(computed_at) AS max_time
                    FROM daily_metrics
                    GROUP BY ticker
                ) latest
                ON dm.ticker = latest.ticker AND dm.computed_at = latest.max_time
            """
            df = pd.read_sql_query(query, conn)
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        return df
    except sqlite3.Error:
        logger.exception("Failed to fetch latest metrics")
        return pd.DataFrame()
