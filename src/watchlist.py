from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import Candidate


SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    confidence TEXT NOT NULL,
    score INTEGER NOT NULL,
    setup TEXT NOT NULL,
    entry REAL NOT NULL,
    stop REAL NOT NULL,
    target1 REAL NOT NULL,
    target2 REAL NOT NULL,
    payload TEXT NOT NULL
)
"""


class WatchlistStore:
    def __init__(self, path: str = "data/watchlist.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upsert(self, candidates: Iterable[Candidate]) -> None:
        sql = """
        INSERT INTO watchlist (
            symbol, name, confidence, score, setup, entry, stop, target1, target2, payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            name=excluded.name,
            updated_at=CURRENT_TIMESTAMP,
            confidence=excluded.confidence,
            score=excluded.score,
            setup=excluded.setup,
            entry=excluded.entry,
            stop=excluded.stop,
            target1=excluded.target1,
            target2=excluded.target2,
            payload=excluded.payload
        """
        rows = [
            (
                c.symbol, c.name, c.confidence, c.score, c.setup,
                c.entry, c.stop, c.target1, c.target2, json.dumps(c.to_dict())
            )
            for c in candidates
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def dataframe(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, name, added_at, updated_at, confidence, score, setup, "
                "entry, stop, target1, target2, payload FROM watchlist "
                "ORDER BY score DESC, updated_at DESC"
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        columns = [
            "symbol", "name", "added_at", "updated_at", "confidence", "score", "setup",
            "entry", "stop", "target1", "target2", "payload"
        ]
        df = pd.DataFrame(rows, columns=columns)
        payloads = df["payload"].map(json.loads)
        df["analysis"] = payloads.map(
            lambda x: x.get("ai_summary") or x.get("rule_summary", "")
        )
        df["Yahoo chart"] = payloads.map(lambda x: x.get("chart_yahoo", ""))
        df["TradingView chart"] = payloads.map(lambda x: x.get("chart_tradingview", ""))
        return df.drop(columns=["payload"])

    def remove(self, symbol: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
