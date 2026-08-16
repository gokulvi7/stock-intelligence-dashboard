"""Stock Intelligence Dashboard - main entry point.

Handles page config, light theme, CSV upload, the end-to-end data pipeline
(Yahoo Finance retrieval -> sentiment analysis -> metrics aggregation ->
SQLite persistence), global sidebar filters, and multipage navigation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from config import APP_ICON, APP_TITLE, UPLOADS_DIR
from database import database
from services import metrics_service, sentiment_service, yahoo_service
from utils.helpers import get_logger, validate_ticker_csv

logger = get_logger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

database.init_db()

# --------------------------------------------------------------------------
# Global CSS - finance-grade light theme polish
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E0E3E7;
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.85; }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .app-hero {
        padding: 0.25rem 0 0.75rem 0;
        border-bottom: 1px solid #E0E3E7;
        margin-bottom: 1rem;
    }

    /* Equal-height, aligned cards: any st.container(border=True, key="card-...")
       stretches to match its tallest sibling within the same row of columns,
       instead of each card sizing to its own content. */
    div[data-testid="stHorizontalBlock"] { align-items: stretch; }
    div[data-testid="stColumn"] { display: flex; }
    div[data-testid="stColumn"] > div { width: 100%; }
    div[class*="st-key-card-"] {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    div[class*="st-key-card-"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        "stocks_df": pd.DataFrame(),
        "metrics_df": pd.DataFrame(),
        "news_df": pd.DataFrame(),
        "price_history": {},
        "companies_map": {},
        "data_loaded": False,
        "last_loaded_at": None,
        "selected_tickers": [],
        "load_warnings": [],
        "load_errors": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# --------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------
def run_pipeline(tickers: List[str], companies_map: Dict[str, str]) -> None:
    """Fetch info/history/news, run sentiment, compute metrics, persist to DB."""

    stocks_rows = []
    metrics_rows = []
    news_rows = []
    price_history: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []

    progress = st.progress(0.0, text="Starting data pipeline...")
    total = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        company_name = companies_map.get(ticker, ticker)
        progress.progress((i - 1) / total, text=f"Fetching {ticker} ({i}/{total})...")

        stock = yahoo_service.fetch_stock_info(ticker, company_name)
        if stock.fetch_error:
            errors.append(f"{ticker}: {stock.fetch_error}")

        history = yahoo_service.fetch_price_history(ticker)
        articles = yahoo_service.fetch_news(ticker)

        progress.progress((i - 0.6) / total, text=f"Analyzing sentiment for {ticker} ({i}/{total})...")
        sentiment_scores = sentiment_service.analyze_news_articles(articles) if articles else []

        metrics = metrics_service.build_daily_metrics(stock, history, sentiment_scores)

        database.upsert_stock(stock)
        database.insert_news_articles(articles)
        database.insert_sentiment_scores(sentiment_scores)
        database.insert_daily_metrics(metrics)

        stocks_rows.append(vars(stock))
        metrics_rows.append(vars(metrics))
        price_history[ticker] = history

        sentiment_by_title = {s.title: s for s in sentiment_scores}
        for article in articles:
            s = sentiment_by_title.get(article.title)
            news_rows.append(
                {
                    "ticker": article.ticker,
                    "title": article.title,
                    "publisher": article.publisher,
                    "link": article.link,
                    "published_at": article.published_at,
                    "label": s.label if s else None,
                    "score": s.score if s else None,
                    "signed_score": s.signed_score if s else None,
                }
            )

    progress.progress(1.0, text="Finalizing...")

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_service.rank_watchlist(metrics_df)

    st.session_state["stocks_df"] = pd.DataFrame(stocks_rows)
    st.session_state["metrics_df"] = metrics_df
    st.session_state["news_df"] = pd.DataFrame(news_rows)
    st.session_state["price_history"] = price_history
    st.session_state["companies_map"] = companies_map
    st.session_state["selected_tickers"] = tickers
    st.session_state["data_loaded"] = True
    st.session_state["last_loaded_at"] = datetime.now(timezone.utc)
    st.session_state["load_errors"] = errors

    progress.empty()

    if errors:
        st.warning(
            f"{len(errors)} ticker(s) had issues while fetching from Yahoo Finance:\n\n"
            + "\n".join(f"- {e}" for e in errors)
        )
    st.success(f"Loaded data for {len(tickers)} ticker(s).")


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon="\U0001F4CA", default=True),
    st.Page("pages/watchlist.py", title="Watchlist", icon="\U0001F4CB"),
    st.Page("pages/stock_detail.py", title="Stock Detail", icon="\U0001F50D"),
    st.Page("pages/ai_insights.py", title="AI Insights", icon="\U0001F916"),
]
pg = st.navigation(pages)

# --------------------------------------------------------------------------
# Sidebar: upload + pipeline controls + global filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.caption("Upload a ticker CSV to begin.")

    uploaded_file = st.file_uploader("Ticker CSV", type=["csv"], help="Columns: ticker, company")

    sample_csv_path = Path(__file__).parent / "sample_data" / "sample_tickers.csv"
    if sample_csv_path.exists():
        st.download_button(
            "Download sample CSV",
            data=sample_csv_path.read_bytes(),
            file_name="sample_tickers.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV file: {exc}")
            raw_df = None

        if raw_df is not None:
            validation = validate_ticker_csv(raw_df)
            for w in validation.warnings:
                st.warning(w)
            for e in validation.errors:
                st.error(e)

            if validation.is_valid:
                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                save_path = UPLOADS_DIR / uploaded_file.name
                try:
                    save_path.write_bytes(uploaded_file.getvalue())
                except OSError:
                    logger.exception("Could not persist uploaded file")

                st.success(f"Found {len(validation.tickers)} valid ticker(s).")
                if st.button("Load / Refresh Data", type="primary", use_container_width=True):
                    with st.spinner("Running data pipeline (Yahoo Finance + sentiment)..."):
                        run_pipeline(validation.tickers, validation.companies)

    st.divider()

    if st.session_state["data_loaded"]:
        metrics_df = st.session_state["metrics_df"]
        all_tickers = sorted(metrics_df["ticker"].unique()) if not metrics_df.empty else []
        st.session_state["selected_tickers"] = all_tickers

        st.session_state["news_lookback_days"] = st.slider(
            "News lookback (days)", min_value=1, max_value=90, value=30, key="news_lookback_slider"
        )

        last_loaded = st.session_state["last_loaded_at"]
        if last_loaded:
            st.caption(f"Last updated: {last_loaded.strftime('%Y-%m-%d %H:%M UTC')}")

        if st.button("Clear Data", use_container_width=True):
            for key in ("stocks_df", "metrics_df", "news_df"):
                st.session_state[key] = pd.DataFrame()
            st.session_state["price_history"] = {}
            st.session_state["data_loaded"] = False
            st.rerun()
    else:
        st.info("No data loaded yet. Upload a CSV and click **Load / Refresh Data**.")

pg.run()
