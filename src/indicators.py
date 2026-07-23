from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def macd_histogram(close: pd.Series) -> pd.Series:
    macd = ema(close, 12) - ema(close, 26)
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd - signal


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    return pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(frame).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rolling_vwap(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    pv = typical * frame["Volume"]
    volume_sum = frame["Volume"].rolling(window, min_periods=window).sum()
    return pv.rolling(window, min_periods=window).sum() / volume_sum.replace(0, np.nan)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA20"] = ema(df["Close"], 20)
    df["RSI14"] = rsi(df["Close"], 14)
    df["MACD_HIST"] = macd_histogram(df["Close"])
    df["ATR14"] = atr(df, 14)
    df["VWAP20"] = rolling_vwap(df, 20)
    df["HIGH20_PREV"] = df["High"].shift(1).rolling(20, min_periods=20).max()
    df["MEDIAN_VOL20_PREV"] = df["Volume"].shift(1).rolling(20, min_periods=20).median()
    df["MEDIAN_DOLLAR20_PREV"] = (
        (df["Close"] * df["Volume"]).shift(1).rolling(20, min_periods=20).median()
    )
    df["REL_VOL"] = df["Volume"] / df["MEDIAN_VOL20_PREV"].replace(0, np.nan)
    df["CHANGE_PCT"] = df["Close"].pct_change() * 100
    return df
