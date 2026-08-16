"""Stock Detail page - deep dive into a single ticker."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import COLOR_ACCENT, COLOR_TEXT, MOVING_AVERAGE_COLORS, PLOTLY_TEMPLATE, SENTIMENT_COLOR_MAP
from utils.helpers import format_currency, format_large_number, format_percent, safe_round

st.title("Stock Detail")

if not st.session_state.get("data_loaded"):
    st.info("Upload a ticker CSV in the sidebar and click **Load / Refresh Data** to get started.")
    st.stop()

metrics_df: pd.DataFrame = st.session_state["metrics_df"]
stocks_df: pd.DataFrame = st.session_state["stocks_df"]
news_df: pd.DataFrame = st.session_state["news_df"]
price_history: dict = st.session_state["price_history"]
selected = st.session_state.get("selected_tickers") or metrics_df["ticker"].tolist()

if not selected:
    st.warning("No tickers selected. Adjust the filter in the sidebar.")
    st.stop()

ticker = st.selectbox("Select a stock", options=sorted(selected))

stock_row = stocks_df[stocks_df["ticker"] == ticker]
metrics_row = metrics_df[metrics_df["ticker"] == ticker]

if stock_row.empty or metrics_row.empty:
    st.error(f"No data available for {ticker}.")
    st.stop()

stock = stock_row.iloc[0]
metrics = metrics_row.iloc[0]

if stock.get("fetch_error"):
    st.warning(f"Yahoo Finance lookup issue for {ticker}: {stock['fetch_error']}")

# --------------------------------------------------------------------------
# Company info
# --------------------------------------------------------------------------
st.subheader(f"{stock['company']} ({ticker})")
info_cols = st.columns(4)
info_cols[0].markdown(f"**Sector**\n\n{stock.get('sector') or 'N/A'}")
info_cols[1].markdown(f"**Industry**\n\n{stock.get('industry') or 'N/A'}")
info_cols[2].markdown(f"**Recommendation**\n\n{stock.get('recommendation') or 'N/A'}")
info_cols[3].markdown(f"**Sentiment Trend**\n\n{metrics.get('sentiment_trend') or 'N/A'}")

st.divider()

# --------------------------------------------------------------------------
# Key metrics
# --------------------------------------------------------------------------
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Price", format_currency(stock.get("current_price"), stock.get("currency") or "USD"))
m2.metric("Market Cap", format_large_number(stock.get("market_cap")))
m3.metric("PE Ratio", f"{stock['pe_ratio']:.2f}" if pd.notna(stock.get("pe_ratio")) else "N/A")
m4.metric("Analyst Target", format_currency(stock.get("analyst_target_price"), stock.get("currency") or "USD"))
m5.metric("52W High / Low",
          f"{format_currency(stock.get('week52_high'))} / {format_currency(stock.get('week52_low'))}")
m6.metric("30D Return", format_percent(metrics.get("return_30d")))

st.divider()

# --------------------------------------------------------------------------
# 1. Price history + 2. Moving averages (combined line chart)
# --------------------------------------------------------------------------
with st.container(border=True, key="card-price-history"):
    st.subheader("Price History & Moving Averages (1 Year)")
    hist = price_history.get(ticker, pd.DataFrame())

    if hist.empty:
        st.caption("No historical price data available.")
    else:
        hist = hist.sort_values("Date").copy()
        for window in (20, 50, 200):
            hist[f"MA{window}"] = hist["Close"].rolling(window=window, min_periods=max(2, window // 4)).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["Close"], mode="lines", name="Close",
            line=dict(color=COLOR_TEXT, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
        ))
        for window in (20, 50, 200):
            fig.add_trace(go.Scatter(
                x=hist["Date"], y=hist[f"MA{window}"], mode="lines", name=f"{window}-Day MA",
                line=dict(color=MOVING_AVERAGE_COLORS[window], width=1.5, dash="dot"),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{window}D MA: $%{{y:.2f}}<extra></extra>",
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# 3. Sentiment trend + 4. News activity trend
# --------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

ticker_news = news_df[news_df["ticker"] == ticker].copy() if not news_df.empty else pd.DataFrame()
if not ticker_news.empty:
    ticker_news["published_at"] = pd.to_datetime(ticker_news["published_at"], errors="coerce")
    ticker_news = ticker_news.dropna(subset=["published_at"]).sort_values("published_at")
    ticker_news["date"] = ticker_news["published_at"].dt.date

with chart_col1, st.container(border=True, key="card-sentiment-trend"):
    st.subheader("Sentiment Trend")
    if ticker_news.empty or ticker_news["signed_score"].isna().all():
        st.caption("No news sentiment history available.")
    else:
        daily_sentiment = ticker_news.groupby("date")["signed_score"].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=daily_sentiment["date"], y=daily_sentiment["signed_score"],
            mode="lines+markers", line=dict(color=COLOR_ACCENT, width=2),
            marker=dict(size=7),
            hovertemplate="%{x}<br>Avg Sentiment: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=1, line_color="#555")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Date",
            yaxis_title="Avg Sentiment",
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

with chart_col2, st.container(border=True, key="card-news-activity"):
    st.subheader("News Activity Trend")
    if ticker_news.empty:
        st.caption("No news activity available.")
    else:
        daily_count = ticker_news.groupby("date").size().reset_index(name="count")
        fig = go.Figure(go.Bar(
            x=daily_count["date"], y=daily_count["count"],
            marker_color=COLOR_ACCENT,
            hovertemplate="%{x}<br>%{y} article(s)<extra></extra>",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Date",
            yaxis_title="Articles",
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# News section
# --------------------------------------------------------------------------
st.subheader("Recent News")
with st.container(height=480, border=True, key="card-recent-news"):
    if ticker_news.empty:
        st.caption("No recent news found for this ticker.")
    else:
        for _, row in ticker_news.sort_values("published_at", ascending=False).iterrows():
            label = (row.get("label") or "neutral").lower()
            color = SENTIMENT_COLOR_MAP.get(label, "#95A5A6")
            date_str = row["published_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["published_at"]) else "N/A"
            score = safe_round(row.get("signed_score"), 2)

            # News titles/links/publishers originate from an external feed, so they
            # are HTML-escaped before being interpolated into unsafe_allow_html markup.
            safe_title = html.escape(str(row["title"]))
            safe_publisher = html.escape(str(row.get("publisher") or ""))
            link = row.get("link")
            if isinstance(link, str) and link.startswith(("http://", "https://")):
                title_html = f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer" style="color:{COLOR_TEXT};">{safe_title}</a>'
            else:
                title_html = safe_title

            st.markdown(
                f"""<div style="border-left:3px solid {color}; padding:6px 12px; margin-bottom:8px;">
                <b>{title_html}</b><br>
                <span style="color:#8B949E;font-size:0.85em;">{safe_publisher} &middot; {date_str}</span><br>
                <span style="color:{color};font-weight:600;">{html.escape(label.title())}</span>
                <span style="color:#8B949E;"> (score: {score if score is not None else 'N/A'})</span>
                </div>""",
                unsafe_allow_html=True,
            )
