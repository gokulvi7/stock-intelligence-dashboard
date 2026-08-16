"""Shared utility helpers: logging, CSV validation, and formatting."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import LOG_LEVEL, MAX_TICKERS_PER_UPLOAD

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root handler once."""

    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)


@dataclass
class CsvValidationResult:
    """Outcome of validating an uploaded ticker CSV."""

    is_valid: bool
    tickers: List[str]
    companies: dict
    errors: List[str]
    warnings: List[str]


def validate_ticker_csv(df: pd.DataFrame) -> CsvValidationResult:
    """Validate an uploaded CSV of tickers.

    Expects at least a ``ticker`` column (case-insensitive) and an optional
    ``company`` column. Rows with missing/blank tickers are dropped with a
    warning rather than failing the whole upload.
    """

    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        return CsvValidationResult(False, [], {}, ["The uploaded CSV is empty."], [])

    normalized_cols = {c.strip().lower(): c for c in df.columns}
    if "ticker" not in normalized_cols:
        errors.append("CSV must contain a 'ticker' column.")
        return CsvValidationResult(False, [], {}, errors, warnings)

    ticker_col = normalized_cols["ticker"]
    company_col = normalized_cols.get("company")

    work = df.copy()
    work[ticker_col] = work[ticker_col].astype(str).str.strip().str.upper()

    missing_mask = work[ticker_col].isin(["", "NAN", "NONE"]) | work[ticker_col].isna()
    n_missing = int(missing_mask.sum())
    if n_missing:
        warnings.append(f"Dropped {n_missing} row(s) with missing ticker values.")
    work = work[~missing_mask]

    before_dedup = len(work)
    work = work.drop_duplicates(subset=[ticker_col])
    if before_dedup != len(work):
        warnings.append(f"Removed {before_dedup - len(work)} duplicate ticker(s).")

    if len(work) > MAX_TICKERS_PER_UPLOAD:
        warnings.append(
            f"CSV contains {len(work)} tickers; only the first "
            f"{MAX_TICKERS_PER_UPLOAD} will be processed to keep the app responsive."
        )
        work = work.head(MAX_TICKERS_PER_UPLOAD)

    if work.empty:
        errors.append("No valid ticker rows remain after cleaning the CSV.")
        return CsvValidationResult(False, [], {}, errors, warnings)

    tickers = work[ticker_col].tolist()
    if company_col:
        companies = dict(zip(work[ticker_col], work[company_col].astype(str)))
    else:
        companies = {t: t for t in tickers}

    return CsvValidationResult(True, tickers, companies, errors, warnings)


def format_currency(value: Optional[float], currency: str = "USD") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{value:,.2f}"


def format_large_number(value: Optional[float]) -> str:
    """Format large numbers (market cap) with T/B/M suffixes."""

    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{suffix}"
    return f"${value:,.2f}"


def format_percent(value: Optional[float], decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.{decimals}f}%"


def sentiment_to_color(label: Optional[str]) -> str:
    from config import SENTIMENT_COLOR_MAP, COLOR_NEUTRAL

    if not label:
        return COLOR_NEUTRAL
    return SENTIMENT_COLOR_MAP.get(label.lower(), COLOR_NEUTRAL)


def safe_round(value: Optional[float], decimals: int = 2) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None
