import numpy as np
import pandas as pd

from src.models import ScanConfig
from src.scanner import analyse_symbol


def synthetic_breakout():
    n = 90
    close = np.linspace(10, 12, n)
    volume = np.full(n, 1_000_000.0)
    close[-1] = 12.8
    volume[-1] = 3_000_000
    frame = pd.DataFrame(
        {
            "Open": close - 0.05,
            "High": close + 0.12,
            "Low": close - 0.12,
            "Close": close,
            "Volume": volume,
        },
        index=pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
    )
    return frame


def test_breakout_candidate_has_valid_levels():
    cfg = ScanConfig(min_median_dollar_volume=1_000_000, min_relative_volume=1.2)
    candidate = analyse_symbol("TEST.AX", "Test", synthetic_breakout(), cfg)
    assert candidate is not None
    assert candidate.entry > candidate.stop
    assert candidate.target1 > candidate.entry
    assert candidate.target2 > candidate.target1
    assert 0 <= candidate.score <= 100
