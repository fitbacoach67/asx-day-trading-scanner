import numpy as np
import pandas as pd

from src.indicators import add_indicators


def test_add_indicators_has_expected_columns():
    n = 80
    close = np.linspace(10, 14, n)
    frame = pd.DataFrame(
        {
            "Open": close - 0.05,
            "High": close + 0.15,
            "Low": close - 0.15,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, n),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
    )
    result = add_indicators(frame)
    expected = {
        "EMA9", "EMA20", "RSI14", "MACD_HIST", "ATR14", "VWAP20",
        "HIGH20_PREV", "MEDIAN_DOLLAR20_PREV", "REL_VOL", "CHANGE_PCT"
    }
    assert expected.issubset(result.columns)
    assert result["ATR14"].dropna().iloc[-1] > 0
