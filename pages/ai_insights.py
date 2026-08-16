"""AI Insights page - local, rule-based investment-style summaries per stock."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE
from services.metrics_service import generate_ai_summary

st.title("AI Insights")
st.caption(
    "Investment-style summaries generated locally from computed sentiment and "
    "price metrics - no external LLM call is made."
)

if not st.session_state.get("data_loaded"):
    st.info("Upload a ticker CSV in the sidebar and click **Load / Refresh Data** to get started.")
    st.stop()

metrics_df: pd.DataFrame = st.session_state["metrics_df"]
stocks_df: pd.DataFrame = st.session_state["stocks_df"]
news_df: pd.DataFrame = st.session_state["news_df"]
selected = st.session_state.get("selected_tickers") or metrics_df["ticker"].tolist()

metrics_df = metrics_df[metrics_df["ticker"].isin(selected)]
if metrics_df.empty:
    st.warning("No tickers selected. Adjust the filter in the sidebar.")
    st.stop()

sort_choice = st.radio(
    "Sort stocks by", ["Watchlist Rank", "Most Bullish", "Most Bearish"], horizontal=True
)
if sort_choice == "Most Bullish":
    ordered = metrics_df.sort_values("bullish_score", ascending=False)
elif sort_choice == "Most Bearish":
    ordered = metrics_df.sort_values("bearish_score", ascending=False)
else:
    ordered = metrics_df.sort_values("watchlist_rank", ascending=True)

STANCE_COLOR = {"bullish": COLOR_POSITIVE, "bearish": COLOR_NEGATIVE}

for _, metrics_row in ordered.iterrows():
    ticker = metrics_row["ticker"]
    stock_match = stocks_df[stocks_df["ticker"] == ticker]
    company = stock_match.iloc[0]["company"] if not stock_match.empty else ticker
    news_subset = news_df[news_df["ticker"] == ticker] if not news_df.empty else pd.DataFrame()

    summary = generate_ai_summary(ticker, company, metrics_row, news_subset)
    stance_color = STANCE_COLOR.get(summary["overall_stance"], COLOR_NEUTRAL)

    with st.container(border=True):
        header_col, badge_col = st.columns([4, 1])
        header_col.markdown(f"### {company} ({ticker})")
        badge_col.markdown(
            f"<div style='text-align:right;'><span style='background:{stance_color};"
            f"color:#0E1117;padding:4px 12px;border-radius:12px;font-weight:700;'>"
            f"{summary['overall_stance'].upper()}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(summary["bullish_summary"])

        if summary["themes"]:
            st.markdown(
                "**Key Discussion Themes:** "
                + " &nbsp; ".join(f"`{t}`" for t in summary["themes"])
            )

        opp_col, risk_col = st.columns(2)
        with opp_col:
            st.markdown("**Key Opportunities**")
            for item in summary["key_opportunities"]:
                st.markdown(f"- {item}")
        with risk_col:
            st.markdown("**Bearish Risks / Key Concerns**")
            for item in summary["bearish_risks"]:
                st.markdown(f"- {item}")

st.divider()
st.caption(
    "Summaries are generated from FinBERT sentiment scores and computed price "
    "momentum using local template logic - they are not investment advice."
)
