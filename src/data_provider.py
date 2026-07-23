from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def history(
        self, symbols: list[str], period: str, interval: str
    ) -> Mapping[str, pd.DataFrame]:
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Prototype data provider. Do not assume it is real-time or execution-grade."""

    def history(
        self, symbols: list[str], period: str, interval: str
    ) -> Mapping[str, pd.DataFrame]:
        if not symbols:
            return {}

        import yfinance as yf

        raw = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
            actions=False,
            repair=True,
            timeout=25,
        )

        result: dict[str, pd.DataFrame] = {}
        if len(symbols) == 1:
            symbol = symbols[0]
            if not raw.empty:
                result[symbol] = self._normalise(raw)
            return result

        if not isinstance(raw.columns, pd.MultiIndex):
            return result

        level0 = set(raw.columns.get_level_values(0))
        for symbol in symbols:
            if symbol in level0:
                frame = raw[symbol].copy()
                if not frame.empty:
                    result[symbol] = self._normalise(frame)
        return result

    @staticmethod
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        expected = ["Open", "High", "Low", "Close", "Volume"]
        df = frame.copy()
        df = df[[c for c in expected if c in df.columns]]
        if len(df.columns) != len(expected):
            return pd.DataFrame(columns=expected)
        df.index = pd.to_datetime(df.index, utc=True)
        for col in expected:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=expected).sort_index()


def age_minutes(timestamp: pd.Timestamp) -> float:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return max(0.0, (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds() / 60)
