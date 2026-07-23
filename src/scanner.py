from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .data_provider import age_minutes
from .indicators import add_indicators
from .models import Candidate, ScanConfig


def _finite(value: float) -> bool:
    return value is not None and math.isfinite(float(value))


def _round_price(value: float) -> float:
    # ASX tick sizes vary. This is an analytical approximation, not an order-price validator.
    if value < 0.10:
        return round(value, 3)
    if value < 2:
        return round(value, 3)
    return round(value, 2)


def _confidence(score: int, cfg: ScanConfig) -> str:
    if score >= cfg.high_threshold:
        return "High"
    if score >= cfg.moderate_threshold:
        return "Moderate"
    return "Low"


def _score(latest: pd.Series, previous: pd.Series) -> tuple[int, list[str], str]:
    score = 0
    reasons: list[str] = []

    rel_vol = float(latest["REL_VOL"])
    if rel_vol >= 3:
        score += 25
        reasons.append(f"exceptional relative volume {rel_vol:.2f}x")
    elif rel_vol >= 2:
        score += 21
        reasons.append(f"strong relative volume {rel_vol:.2f}x")
    elif rel_vol >= 1.5:
        score += 16
        reasons.append(f"elevated relative volume {rel_vol:.2f}x")
    elif rel_vol >= 1.25:
        score += 10
        reasons.append(f"notable relative volume {rel_vol:.2f}x")

    close = float(latest["Close"])
    ema9 = float(latest["EMA9"])
    ema20 = float(latest["EMA20"])
    if close > ema9 > ema20:
        score += 18
        reasons.append("price above aligned EMA 9/20")
    elif close > ema20:
        score += 9
        reasons.append("price above EMA20")

    if ema9 > ema20 and float(previous["EMA9"]) <= float(previous["EMA20"]):
        score += 8
        reasons.append("fresh EMA9/EMA20 bullish cross")

    rsi = float(latest["RSI14"])
    if 55 <= rsi <= 68:
        score += 13
        reasons.append(f"constructive RSI {rsi:.1f}")
    elif 50 <= rsi < 55 or 68 < rsi <= 74:
        score += 7
        reasons.append(f"positive RSI {rsi:.1f}")
    elif rsi > 78:
        score -= 7
        reasons.append(f"overextended RSI {rsi:.1f}")

    macd = float(latest["MACD_HIST"])
    prev_macd = float(previous["MACD_HIST"])
    if macd > 0 and macd > prev_macd:
        score += 11
        reasons.append("positive and rising MACD histogram")
    elif macd > 0:
        score += 6
        reasons.append("positive MACD histogram")

    vwap = float(latest["VWAP20"])
    if close > vwap:
        score += 8
        reasons.append("price above rolling VWAP")

    breakout = float(latest["HIGH20_PREV"])
    atr = float(latest["ATR14"])
    distance_atr = (breakout - close) / atr if atr > 0 else np.nan
    if close >= breakout:
        score += 18
        reasons.append("20-bar breakout")
        setup = "Breakout"
    elif _finite(distance_atr) and 0 <= distance_atr <= 0.35:
        score += 11
        reasons.append("within 0.35 ATR of 20-bar breakout")
        setup = "Breakout watch"
    elif close > ema9 > ema20 and abs(close - ema9) <= 0.45 * atr:
        score += 8
        reasons.append("trend pullback near EMA9")
        setup = "Trend pullback"
    else:
        setup = "Momentum continuation"

    change_pct = float(latest["CHANGE_PCT"])
    if 0.5 <= change_pct <= 4.5:
        score += 8
        reasons.append(f"controlled positive move {change_pct:.2f}%")
    elif change_pct > 8:
        score -= 10
        reasons.append(f"large one-bar move {change_pct:.2f}% increases chase risk")
    elif change_pct < -1:
        score -= 8
        reasons.append(f"negative latest move {change_pct:.2f}%")

    return max(0, min(100, int(round(score)))), reasons, setup


def _trade_levels(latest: pd.Series, setup: str) -> tuple[float, float, float, float, float]:
    close = float(latest["Close"])
    high = float(latest["High"])
    low = float(latest["Low"])
    atr = float(latest["ATR14"])
    breakout = float(latest["HIGH20_PREV"])
    ema9 = float(latest["EMA9"])

    if setup in {"Breakout", "Breakout watch"}:
        entry = max(close, breakout + 0.05 * atr)
        structural_stop = min(low, breakout - 0.55 * atr)
    else:
        entry = close + 0.10 * atr
        structural_stop = min(low, ema9 - 0.60 * atr)

    stop = min(entry - 0.80 * atr, structural_stop)
    risk = entry - stop
    if risk <= 0:
        risk = max(atr, entry * 0.01)
        stop = entry - risk

    target1 = entry + 1.5 * risk
    target2 = entry + 2.5 * risk
    rr1 = (target1 - entry) / risk
    return tuple(_round_price(x) for x in (entry, stop, target1, target2)) + (round(rr1, 2),)


def analyse_symbol(
    symbol: str,
    name: str,
    frame: pd.DataFrame,
    cfg: ScanConfig,
) -> Candidate | None:
    if frame is None or len(frame) < 45:
        return None

    df = add_indicators(frame)
    needed = [
        "EMA9", "EMA20", "RSI14", "MACD_HIST", "ATR14", "VWAP20",
        "HIGH20_PREV", "MEDIAN_DOLLAR20_PREV", "REL_VOL", "CHANGE_PCT"
    ]
    df = df.dropna(subset=needed)
    if len(df) < 2:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    close = float(latest["Close"])
    median_dollar = float(latest["MEDIAN_DOLLAR20_PREV"])
    rel_vol = float(latest["REL_VOL"])

    if close < cfg.min_price:
        return None
    if median_dollar < cfg.min_median_dollar_volume:
        return None
    if rel_vol < cfg.min_relative_volume:
        return None

    score, reasons, setup = _score(latest, previous)
    entry, stop, target1, target2, rr1 = _trade_levels(latest, setup)
    confidence = _confidence(score, cfg)
    ticker_code = symbol.removesuffix(".AX")
    timestamp = pd.Timestamp(df.index[-1])

    return Candidate(
        symbol=symbol,
        name=name,
        timestamp=timestamp.isoformat(),
        data_age_minutes=round(age_minutes(timestamp), 1),
        close=_round_price(close),
        volume=float(latest["Volume"]),
        median_dollar_volume=round(median_dollar, 0),
        relative_volume=round(rel_vol, 2),
        change_pct=round(float(latest["CHANGE_PCT"]), 2),
        ema9=_round_price(float(latest["EMA9"])),
        ema20=_round_price(float(latest["EMA20"])),
        rsi14=round(float(latest["RSI14"]), 1),
        macd_hist=round(float(latest["MACD_HIST"]), 4),
        atr14=_round_price(float(latest["ATR14"])),
        vwap=_round_price(float(latest["VWAP20"])),
        breakout_20=_round_price(float(latest["HIGH20_PREV"])),
        score=score,
        confidence=confidence,
        setup=setup,
        entry=entry,
        stop=stop,
        target1=target1,
        target2=target2,
        reward_risk_t1=rr1,
        chart_yahoo=f"https://finance.yahoo.com/quote/{symbol}/chart/",
        chart_tradingview=f"https://www.tradingview.com/chart/?symbol=ASX%3A{ticker_code}",
        rule_summary="; ".join(reasons),
    )


def scan_market(
    histories: Mapping[str, pd.DataFrame],
    names: Mapping[str, str],
    cfg: ScanConfig,
) -> list[Candidate]:
    candidates = []
    for symbol, frame in histories.items():
        try:
            candidate = analyse_symbol(symbol, names.get(symbol, symbol), frame, cfg)
            if candidate:
                candidates.append(candidate)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
    return sorted(
        candidates,
        key=lambda c: (c.score, c.relative_volume, c.median_dollar_volume),
        reverse=True,
    )
