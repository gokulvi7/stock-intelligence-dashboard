"""Central configuration and constants for the Stock Intelligence Dashboard.

Loads environment variables via python-dotenv and exposes them as typed
module-level constants so the rest of the app never touches os.environ
directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(CACHE_DIR / "stock_intelligence.db")))

for _dir in (UPLOADS_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# App behaviour
# --------------------------------------------------------------------------
APP_TITLE = "Stock Intelligence Dashboard"
APP_ICON = "\U0001F4C8"  # chart increasing emoji

PRICE_HISTORY_PERIOD = os.getenv("PRICE_HISTORY_PERIOD", "1y")
PRICE_CACHE_TTL_SECONDS = int(os.getenv("PRICE_CACHE_TTL_SECONDS", "900"))
INFO_CACHE_TTL_SECONDS = int(os.getenv("INFO_CACHE_TTL_SECONDS", "900"))
NEWS_CACHE_TTL_SECONDS = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "900"))
MAX_NEWS_PER_TICKER = int(os.getenv("MAX_NEWS_PER_TICKER", "15"))
MAX_TICKERS_PER_UPLOAD = int(os.getenv("MAX_TICKERS_PER_UPLOAD", "50"))

FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")

# --------------------------------------------------------------------------
# Color scheme (light theme, finance-grade)
# --------------------------------------------------------------------------
COLOR_POSITIVE = "#1E8E3E"   # green
COLOR_NEGATIVE = "#D93025"   # red
COLOR_NEUTRAL = "#5F6368"    # gray
COLOR_ACCENT = "#1A73E8"     # blue accent
COLOR_BACKGROUND = "#FFFFFF"
COLOR_SURFACE = "#F8F9FA"
COLOR_TEXT = "#1F2328"
COLOR_MUTED = "#5F6368"

PLOTLY_TEMPLATE = "plotly_white"

SENTIMENT_COLOR_MAP = {
    "positive": COLOR_POSITIVE,
    "negative": COLOR_NEGATIVE,
    "neutral": COLOR_NEUTRAL,
}

# Diverging colorscale (negative -> neutral gray -> positive) used for
# returns/sentiment heatmaps and the market-cap treemap.
DIVERGING_COLORSCALE = [
    [0.0, COLOR_NEGATIVE],
    [0.5, "#E8EAED"],
    [1.0, COLOR_POSITIVE],
]

# Ordered sequential ramp (short -> long lookback) for moving-average lines.
MOVING_AVERAGE_COLORS = {20: "#7FB8E0", 50: "#3E92CC", 200: "#1B4F72"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
