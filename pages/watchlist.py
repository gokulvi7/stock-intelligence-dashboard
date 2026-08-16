"""Watchlist page - searchable, sortable, filterable table of all tracked stocks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.helpers import format_large_number

st.title("Watchlist")

if not st.session_state.get("data_loaded"):
    st.info("Upload a ticker CSV in the sidebar and click **Load / Refresh Data** to get started.")
    st.stop()

metrics_df: pd.DataFrame = st.session_state["metrics_df"]
stocks_df: pd.DataFrame = st.session_state["stocks_df"]
selected = st.session_state.get("selected_tickers") or metrics_df["ticker"].tolist()

table = metrics_df.merge(
    stocks_df[["ticker", "company", "current_price", "market_cap", "recommendation", "pe_ratio",
               "week52_high", "week52_low", "analyst_target_price"]],
    on="ticker", how="left", suffixes=("", "_stock"),
)
table = table[table["ticker"].isin(selected)].copy()

if table.empty:
    st.warning("No tickers selected. Adjust the filter in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# Search + filters
# --------------------------------------------------------------------------
f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
with f1:
    search = st.text_input("Search ticker or company", placeholder="e.g. NVDA or Nvidia")
with f2:
    recs = ["All"] + sorted(table["recommendation"].dropna().unique().tolist())
    rec_filter = st.selectbox("Recommendation", recs)
with f3:
    trend_options = ["All"] + sorted(table["sentiment_trend"].dropna().unique().tolist())
    trend_filter = st.selectbox("Sentiment Trend", trend_options)
with f4:
    sort_options = {
        "Watchlist Rank": "watchlist_rank",
        "Price": "current_price",
        "Market Cap": "market_cap",
        "Sentiment": "avg_sentiment",
        "30D Return": "return_30d",
    }
    sort_label = st.selectbox("Sort by", list(sort_options.keys()))

filtered = table.copy()
if search:
    mask = (
        filtered["ticker"].str.contains(search, case=False, na=False, regex=False)
        | filtered["company"].str.contains(search, case=False, na=False, regex=False)
    )
    filtered = filtered[mask]
if rec_filter != "All":
    filtered = filtered[filtered["recommendation"] == rec_filter]
if trend_filter != "All":
    filtered = filtered[filtered["sentiment_trend"] == trend_filter]

filtered = filtered.sort_values(sort_options[sort_label], ascending=(sort_label == "Watchlist Rank"))

st.caption(f"Showing {len(filtered)} of {len(table)} tracked stocks.")

display_cols = {
    "watchlist_rank": "Rank",
    "ticker": "Ticker",
    "company": "Company",
    "current_price": "Price",
    "market_cap": "Market Cap",
    "avg_sentiment": "Sentiment",
    "sentiment_trend": "Trend",
    "recommendation": "Recommendation",
    "return_30d": "30D Return %",
}
display = filtered[list(display_cols.keys())].rename(columns=display_cols)
display["Market Cap"] = display["Market Cap"].map(format_large_number)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Sentiment": st.column_config.ProgressColumn(
            min_value=-1.0, max_value=1.0, format="%.2f"
        ),
        "30D Return %": st.column_config.NumberColumn(format="%.2f%%"),
    },
)

st.download_button(
    "Download Watchlist (CSV)",
    data=display.to_csv(index=False).encode("utf-8"),
    file_name="watchlist.csv",
    mime="text/csv",
)

with st.expander("Key metrics detail"):
    detail_cols = ["ticker", "pe_ratio", "week52_high", "week52_low", "analyst_target_price",
                    "return_1d", "return_7d", "return_30d", "return_90d", "news_volume",
                    "bullish_score", "bearish_score"]
    st.dataframe(filtered[detail_cols], use_container_width=True, hide_index=True)
