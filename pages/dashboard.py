"""Overview Dashboard - portfolio-wide KPIs and visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    DIVERGING_COLORSCALE,
    PLOTLY_TEMPLATE,
    SENTIMENT_COLOR_MAP,
)
from utils.helpers import format_large_number, format_percent

st.title("Stock Research and Sentiment Dashboard")

if not st.session_state.get("data_loaded"):
    st.info("Upload a ticker CSV in the sidebar and click **Load / Refresh Data** to get started.")
    st.stop()

metrics_df: pd.DataFrame = st.session_state["metrics_df"]
stocks_df: pd.DataFrame = st.session_state["stocks_df"]
news_df: pd.DataFrame = st.session_state["news_df"]
selected = st.session_state.get("selected_tickers") or metrics_df["ticker"].tolist()

metrics_df = metrics_df[metrics_df["ticker"].isin(selected)].copy()
stocks_df = stocks_df[stocks_df["ticker"].isin(selected)].copy()
news_df = news_df[news_df["ticker"].isin(selected)].copy() if not news_df.empty else news_df

if metrics_df.empty:
    st.warning("No tickers selected. Adjust the filter in the sidebar.")
    st.stop()

merged = metrics_df.merge(stocks_df[["ticker", "company"]], on="ticker", how="left")

# --------------------------------------------------------------------------
# KPI cards
# --------------------------------------------------------------------------
n_stocks = merged["ticker"].nunique()
avg_sentiment = merged["avg_sentiment"].mean()
most_bullish = merged.loc[merged["bullish_score"].idxmax()] if merged["bullish_score"].notna().any() else None
most_bearish = merged.loc[merged["bearish_score"].idxmax()] if merged["bearish_score"].notna().any() else None
best_30d = merged.loc[merged["return_30d"].idxmax()] if merged["return_30d"].notna().any() else None
worst_30d = merged.loc[merged["return_30d"].idxmin()] if merged["return_30d"].notna().any() else None

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stocks Tracked", n_stocks)
k2.metric("Avg Portfolio Sentiment", f"{avg_sentiment:+.2f}" if pd.notna(avg_sentiment) else "N/A")
k3.metric("Most Bullish", most_bullish["ticker"] if most_bullish is not None else "N/A",
          f"score {most_bullish['bullish_score']:.0f}" if most_bullish is not None else None)
k4.metric("Most Bearish", most_bearish["ticker"] if most_bearish is not None else "N/A",
          f"score {most_bearish['bearish_score']:.0f}" if most_bearish is not None else None)
k5.metric("Best 30D Performer", best_30d["ticker"] if best_30d is not None else "N/A",
          format_percent(best_30d["return_30d"]) if best_30d is not None else None)
k6.metric("Worst 30D Performer", worst_30d["ticker"] if worst_30d is not None else "N/A",
          format_percent(worst_30d["return_30d"]) if worst_30d is not None else None)

st.divider()

row1_left, row1_right = st.columns(2)

# --------------------------------------------------------------------------
# 1. Stock Performance Leaderboard (horizontal bar)
# --------------------------------------------------------------------------
with row1_left, st.container(border=True, key="card-perf-leaderboard"):
    st.subheader("Stock Performance (30D Return)")
    perf = merged.dropna(subset=["return_30d"]).sort_values("return_30d")
    if perf.empty:
        st.caption("No return data available.")
    else:
        colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in perf["return_30d"]]
        fig = go.Figure(
            go.Bar(
                x=perf["return_30d"],
                y=perf["ticker"],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>30D Return: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="30-Day Return (%)",
            yaxis_title="",
            height=max(320, 28 * len(perf)),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.add_vline(x=0, line_width=1, line_color=COLOR_NEUTRAL)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View as table"):
            st.dataframe(perf[["ticker", "company", "return_30d"]], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# 2. Sentiment Leaderboard
# --------------------------------------------------------------------------
with row1_right, st.container(border=True, key="card-sentiment-leaderboard"):
    st.subheader("Sentiment Leaderboard")
    sent = merged.dropna(subset=["avg_sentiment"]).sort_values("avg_sentiment")
    if sent.empty:
        st.caption("No sentiment data available.")
    else:
        colors = [
            COLOR_POSITIVE if v > 0.1 else COLOR_NEGATIVE if v < -0.1 else COLOR_NEUTRAL
            for v in sent["avg_sentiment"]
        ]
        fig = go.Figure(
            go.Bar(
                x=sent["avg_sentiment"],
                y=sent["ticker"],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Avg Sentiment: %{x:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Average Sentiment Score (-1 to 1)",
            yaxis_title="",
            height=max(320, 28 * len(sent)),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.add_vline(x=0, line_width=1, line_color=COLOR_NEUTRAL)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View as table"):
            st.dataframe(sent[["ticker", "company", "avg_sentiment"]], use_container_width=True, hide_index=True)

row2_left, row2_right = st.columns(2)

# --------------------------------------------------------------------------
# 3. Market Cap Comparison (treemap, colored by sentiment)
# --------------------------------------------------------------------------
with row2_left, st.container(border=True, key="card-market-cap"):
    st.subheader("Market Cap Comparison", help="Tile size = market cap, color = average sentiment.")
    cap = merged.dropna(subset=["market_cap"])
    if cap.empty:
        fig = go.Figure()
        fig.add_annotation(text="No market cap data available.", showarrow=False, font=dict(color=COLOR_MUTED))
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=10, b=10), height=420,
                           xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.treemap(
            cap,
            path=[px.Constant("Portfolio"), "ticker"],
            values="market_cap",
            color="avg_sentiment",
            color_continuous_scale=DIVERGING_COLORSCALE,
            color_continuous_midpoint=0,
            hover_data={"company": True, "market_cap": ":,.0f", "avg_sentiment": ":.2f"},
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Market Cap: $%{value:,.0f}<br>Sentiment: %{color:.2f}<extra></extra>"
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View as table"):
            st.dataframe(
                cap[["ticker", "company", "market_cap", "avg_sentiment"]].assign(
                    market_cap=lambda d: d["market_cap"].map(format_large_number)
                ),
                use_container_width=True,
                hide_index=True,
            )

# --------------------------------------------------------------------------
# 4. Sentiment Distribution (pie)
# --------------------------------------------------------------------------
with row2_right, st.container(border=True, key="card-sentiment-distribution"):
    st.subheader("News Sentiment Distribution")
    if news_df.empty or news_df["label"].isna().all():
        st.caption("No news sentiment data available.")
    else:
        counts = news_df["label"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
        fig = go.Figure(
            go.Pie(
                labels=[l.title() for l in counts.index],
                values=counts.values,
                marker=dict(colors=[SENTIMENT_COLOR_MAP[l] for l in counts.index]),
                hole=0.45,
                hovertemplate="<b>%{label}</b><br>%{value} articles (%{percent})<extra></extra>",
            )
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View as table"):
            st.dataframe(
                counts.rename("count").rename_axis("label").reset_index(),
                use_container_width=True,
                hide_index=True,
            )

st.divider()

row3_left, row3_right = st.columns([1.2, 1])

# --------------------------------------------------------------------------
# 5. Return Heatmap
# --------------------------------------------------------------------------
with row3_left, st.container(border=True, key="card-return-heatmap"):
    st.subheader("Return Heatmap")
    ret_cols = ["return_1d", "return_7d", "return_30d", "return_90d"]
    heat = merged.set_index("ticker")[ret_cols].rename(
        columns={"return_1d": "1D", "return_7d": "7D", "return_30d": "30D", "return_90d": "90D"}
    )
    if heat.dropna(how="all").empty:
        st.caption("No return data available.")
    else:
        max_abs = float(heat.abs().max().max()) or 1.0
        fig = go.Figure(
            go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorscale=DIVERGING_COLORSCALE,
                zmid=0,
                zmin=-max_abs,
                zmax=max_abs,
                colorbar=dict(title="Return %"),
                hovertemplate="<b>%{y}</b> | %{x}<br>Return: %{z:.2f}%<extra></extra>",
                text=heat.round(1).values,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(320, 32 * len(heat)),
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# 6. Performance vs Sentiment Scatter
# --------------------------------------------------------------------------
with row3_right, st.container(border=True, key="card-performance-vs-sentiment"):
    st.subheader("Performance vs Sentiment")
    scatter = merged.dropna(subset=["return_30d", "avg_sentiment"])
    if scatter.empty:
        st.caption("Not enough data for a scatter plot.")
    else:
        sizes = scatter["market_cap"].fillna(scatter["market_cap"].median() if scatter["market_cap"].notna().any() else 1)
        fig = px.scatter(
            scatter,
            x="return_30d",
            y="avg_sentiment",
            size=sizes,
            color="avg_sentiment",
            color_continuous_scale=DIVERGING_COLORSCALE,
            color_continuous_midpoint=0,
            text="ticker",
            hover_data={"company": True, "return_30d": ":.2f", "avg_sentiment": ":.2f"},
        )
        fig.update_traces(textposition="top center")
        fig.add_hline(y=0, line_width=1, line_color=COLOR_NEUTRAL)
        fig.add_vline(x=0, line_width=1, line_color=COLOR_NEUTRAL)
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="30-Day Price Return (%)",
            yaxis_title="Average News Sentiment",
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(320, 32 * len(heat)) if not heat.dropna(how="all").empty else 420,
        )
        st.plotly_chart(fig, use_container_width=True)
