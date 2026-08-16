# Stock Intelligence Dashboard

An interactive Streamlit analytics platform that turns a CSV of stock tickers
into a full investor dashboard: live Yahoo Finance fundamentals and price
history, FinBERT-powered news sentiment analysis, computed performance/
sentiment metrics, and local AI-generated investment summaries.

## Features

- **CSV upload** — bring your own watchlist (`ticker,company`)
- **Yahoo Finance data** — current price, market cap, PE ratio, 52-week
  high/low, analyst target price, recommendation, 1-year price history, news
- **FinBERT sentiment analysis** — every news headline is scored positive /
  negative / neutral, with an automatic offline fallback if the model can't
  be downloaded
- **Computed metrics** — 1D/7D/30D/90D returns, average sentiment, sentiment
  trend, news volume, bullish/bearish scores, watchlist rank
- **Dashboard** — KPI cards + 6 Plotly visualizations (leaderboards, treemap,
  pie, heatmap, scatter), laid out as equal-height, aligned cards
- **Watchlist** — searchable, sortable, filterable table with CSV export
- **Sidebar ticker search** — type a ticker symbol or company name to filter
  the loaded watchlist
- **Stock Detail** — price + moving averages (20/50/200D), sentiment trend,
  news activity trend, and a news feed
- **AI Insights** — local, rule-based bullish/bearish summaries per stock
- **SQLite persistence** — stocks, news_articles, sentiment_scores,
  daily_metrics tables, schema created automatically
- **Light, finance-grade UI** — green/red/gray semantic color system

## Project Structure

```
stock_intelligence_dashboard/
├── app.py                     # Entry point: config, upload, pipeline, nav
├── config.py                  # Central configuration & constants
├── pages/
│   ├── dashboard.py            # Overview KPIs + visualizations
│   ├── stock_detail.py         # Per-stock deep dive
│   ├── watchlist.py            # Searchable/sortable watchlist table
│   └── ai_insights.py          # Local AI-generated summaries
├── services/
│   ├── yahoo_service.py        # yfinance data retrieval (cached)
│   ├── sentiment_service.py    # FinBERT sentiment analysis + fallback
│   └── metrics_service.py      # Returns, sentiment aggregation, AI summaries
├── database/
│   └── database.py             # SQLite schema + CRUD helpers
├── models/
│   └── stock.py                 # Typed dataclasses (Stock, NewsArticle, ...)
├── utils/
│   └── helpers.py               # Logging, CSV validation, formatting
├── data/
│   ├── uploads/                 # Saved copies of uploaded CSVs
│   └── cache/                   # SQLite database file
├── sample_data/
│   └── sample_tickers.csv       # Example CSV to try the app with
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

1. **Clone / copy the project**, then create a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > Note: `torch` + `transformers` (for FinBERT) are sizeable downloads.
   > If you don't need FinBERT, the app still runs — sentiment analysis
   > automatically falls back to a lightweight local lexicon scorer.

3. **Configure environment variables (optional):**

   ```bash
   cp .env.example .env
   ```

   Defaults work out of the box; edit `.env` to change cache TTLs, the
   FinBERT model id, or the SQLite path.

## Running the App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

### Using the app

1. In the sidebar, upload a CSV with `ticker` (required) and `company`
   (optional) columns — or download the bundled sample CSV to try it out.
2. Click **Load / Refresh Data**. A progress bar tracks fetching Yahoo
   Finance data and running sentiment analysis for each ticker.
3. Use the sidebar search (ticker symbol or company name) and news lookback
   slider to scope the view, then navigate between **Dashboard**,
   **Watchlist**, **Stock Detail**, and **AI Insights** in the left
   navigation.

## CSV Format

```csv
ticker,company
NVDA,Nvidia
TSLA,Tesla
PLTR,Palantir
AAPL,Apple
MSFT,Microsoft
```

Only `ticker` is required; missing/blank tickers are dropped with a
warning, duplicates are removed, and invalid symbols are reported per-row
after the Yahoo Finance lookup instead of failing the whole batch.

## Database Schema

Created automatically on first run at `data/cache/stock_intelligence.db`:

- **stocks** — latest fundamentals per ticker (upserted)
- **news_articles** — deduplicated news headlines per ticker
- **sentiment_scores** — FinBERT label/score per headline
- **daily_metrics** — one row per computed metrics run (historized)

## Error Handling

- Invalid/empty CSV, missing ticker column → inline validation errors
- Missing ticker values / duplicates → dropped with a sidebar warning
- Failed Yahoo Finance lookups → per-ticker error captured, pipeline
  continues for the rest of the batch
- Empty news results → charts/sections show a friendly "no data" message
  instead of crashing
- FinBERT unavailable (no network / not installed) → automatic fallback to
  a local lexicon-based sentiment scorer

## Tech Stack

Streamlit · Plotly · Pandas · NumPy · yfinance · HuggingFace Transformers
(FinBERT) · SQLite · python-dotenv
