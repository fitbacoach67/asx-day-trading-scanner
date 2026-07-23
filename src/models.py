from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScanConfig:
    interval: str = "1d"
    period: str = "6mo"
    min_price: float = 0.50
    min_median_dollar_volume: float = 5_000_000
    min_relative_volume: float = 1.25
    moderate_threshold: int = 65
    high_threshold: int = 78
    max_symbols: int = 100


@dataclass
class Candidate:
    symbol: str
    name: str
    timestamp: str
    data_age_minutes: float | None
    close: float
    volume: float
    median_dollar_volume: float
    relative_volume: float
    change_pct: float
    ema9: float
    ema20: float
    rsi14: float
    macd_hist: float
    atr14: float
    vwap: float
    breakout_20: float
    score: int
    confidence: str
    setup: str
    entry: float
    stop: float
    target1: float
    target2: float
    reward_risk_t1: float
    chart_yahoo: str
    chart_tradingview: str
    rule_summary: str
    ai_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
