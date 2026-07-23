from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.data_provider import YahooFinanceProvider
from src.groq_analysis import available as groq_available
from src.groq_analysis import enrich_candidates
from src.indicators import add_indicators
from src.models import ScanConfig
from src.scanner import scan_market
from src.watchlist import WatchlistStore


load_dotenv()
st.set_page_config(page_title="ASX Trading Scanner", layout="wide")
st.title("ASX short-term trading scanner")
st.caption(
    "Research tool only. Default Yahoo data may be delayed or incomplete. "
    "No orders are placed and every signal requires independent validation."
)


@st.cache_data(ttl=300, show_spinner=False)
def load_universe() -> pd.DataFrame:
    return pd.read_csv("config/asx_liquid.csv").drop_duplicates("symbol")


@st.cache_data(ttl=180, show_spinner=False)
def fetch_data(symbols: tuple[str, ...], period: str, interval: str):
    return YahooFinanceProvider().history(list(symbols), period, interval)


def make_chart(frame: pd.DataFrame, symbol: str) -> go.Figure:
    df = add_indicators(frame).tail(120)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
    fig.update_layout(height=540, xaxis_rangeslider_visible=False)
    return fig


universe = load_universe()
store = WatchlistStore()

with st.sidebar:
    st.header("Scan configuration")
    interval = st.selectbox("Bar interval", ["1d", "60m", "30m", "15m"], index=0)
    period_options = {
        "1d": ["3mo", "6mo", "1y"],
        "60m": ["1mo", "3mo", "6mo"],
        "30m": ["1mo", "3mo"],
        "15m": ["1mo"],
    }
    period = st.selectbox("History", period_options[interval], index=0)
    min_dollar_m = st.number_input(
        "Minimum median dollar turnover ($m)",
        min_value=1.0, max_value=200.0, value=5.0, step=1.0
    )
    min_rel_vol = st.number_input(
        "Minimum relative volume", min_value=1.0, max_value=10.0, value=1.25, step=0.05
    )
    min_price = st.number_input(
        "Minimum share price", min_value=0.01, max_value=100.0, value=0.50, step=0.10
    )
    max_symbols = st.slider(
        "Universe size", min_value=10, max_value=len(universe), value=len(universe)
    )
    use_groq = st.checkbox("Generate Groq summaries", value=groq_available())
    run_scan = st.button("Run scan", type="primary", use_container_width=True)

tabs = st.tabs(["Market scan", "Watchlist", "Method and safeguards"])

with tabs[0]:
    if run_scan:
        cfg = ScanConfig(
            interval=interval,
            period=period,
            min_price=float(min_price),
            min_median_dollar_volume=float(min_dollar_m) * 1_000_000,
            min_relative_volume=float(min_rel_vol),
            max_symbols=max_symbols,
        )
        selected = universe.head(max_symbols)
        symbols = tuple(selected["symbol"].tolist())
        names = dict(zip(selected["symbol"], selected["name"]))

        with st.spinner("Downloading bars and calculating signals..."):
            histories = fetch_data(symbols, period, interval)
            candidates = scan_market(histories, names, cfg)
            qualified = [c for c in candidates if c.confidence in {"Moderate", "High"}]
            if use_groq and qualified:
                enrich_candidates(qualified, limit=12)
            store.upsert(qualified)

        st.session_state["histories"] = histories
        st.session_state["candidates"] = [c.to_dict() for c in candidates]
        st.success(
            f"Scanned {len(histories)} symbols; found {len(candidates)} notable-volume "
            f"shares and {len(qualified)} moderate/high-confidence setups."
        )

    rows = st.session_state.get("candidates", [])
    if rows:
        df = pd.DataFrame(rows)
        display_cols = [
            "symbol", "name", "confidence", "score", "setup", "close",
            "relative_volume", "change_pct", "rsi14", "entry", "stop",
            "target1", "target2", "data_age_minutes", "chart_tradingview"
        ]
        st.dataframe(
            df[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "chart_tradingview": st.column_config.LinkColumn(
                    "Chart", display_text="Open chart"
                ),
                "median_dollar_volume": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

        selected_symbol = st.selectbox("Inspect candidate", df["symbol"].tolist())
        selected_row = df.loc[df["symbol"] == selected_symbol].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{selected_row['score']}/100")
        c2.metric("Relative volume", f"{selected_row['relative_volume']:.2f}x")
        c3.metric("Entry / stop", f"${selected_row['entry']} / ${selected_row['stop']}")
        c4.metric("Targets", f"${selected_row['target1']} / ${selected_row['target2']}")
        st.write(selected_row.get("ai_summary") or selected_row["rule_summary"])

        history = st.session_state.get("histories", {}).get(selected_symbol)
        if history is not None and not history.empty:
            st.plotly_chart(make_chart(history, selected_symbol), use_container_width=True)
    else:
        st.info("Configure the scan and press **Run scan**.")

with tabs[1]:
    watch = store.dataframe()
    if watch.empty:
        st.info("No moderate/high-confidence candidates are currently stored.")
    else:
        st.dataframe(
            watch,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Yahoo chart": st.column_config.LinkColumn(
                    "Yahoo chart", display_text="Open"
                ),
                "TradingView chart": st.column_config.LinkColumn(
                    "TradingView chart", display_text="Open"
                ),
                "analysis": st.column_config.TextColumn(width="large"),
            },
        )
        remove_symbol = st.selectbox("Remove from watchlist", [""] + watch["symbol"].tolist())
        if remove_symbol and st.button("Remove selected"):
            store.remove(remove_symbol)
            st.rerun()

with tabs[2]:
    st.markdown(
        """
### How the decision is made

The market-data and indicator engine performs the decision. Groq only explains the
pre-calculated facts. This prevents an LLM response from becoming the signal source.

### Entry and exits

- Breakout setups enter just above the prior 20-bar high or current close.
- Pullback setups enter slightly above the current close.
- The stop is based on ATR and nearby structure.
- Target 1 is approximately 1.5R.
- Target 2 is approximately 2.5R.

These are candidate levels, not guaranteed fills. Before acting, check spread, depth,
announcements, halts, market trend, news, slippage and total portfolio risk.

### Data warning

For intraday bars, inspect **data_age_minutes**. Stale data invalidates a day-trading signal.
The included provider is deliberately replaceable with a licensed feed.
        """
    )
