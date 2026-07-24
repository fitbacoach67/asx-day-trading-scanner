class YahooFinanceProvider(MarketDataProvider):
    """Prototype data provider. Do not assume it is real-time or execution-grade."""

    def history(
        self,
        symbols: list[str],
        period: str,
        interval: str,
    ) -> Mapping[str, pd.DataFrame]:
        if not symbols:
            return {}

        import yfinance as yf

        batch_size = 50
        result: dict[str, pd.DataFrame] = {}

        for start in range(0, len(symbols), batch_size):
            batch = symbols[start:start + batch_size]

            try:
                raw = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                    actions=False,
                    repair=True,
                    timeout=30,
                )
            except Exception as exc:
                print(
                    f"Yahoo download failed for batch "
                    f"{start + 1}-{start + len(batch)}: {exc}"
                )
                continue

            if raw is None or raw.empty:
                print(
                    f"No Yahoo data returned for batch "
                    f"{start + 1}-{start + len(batch)}"
                )
                continue
        if len(batch) == 1:
            symbol = batch[0]
            result[symbol] = self._normalise(raw)
            continue

        if not isinstance(raw.columns, pd.MultiIndex):
            print(
                f"Unexpected Yahoo column format for batch "
                f"{start + 1}-{start + len(batch)}"
            )
            continue

        level0 = set(raw.columns.get_level_values(0))

        for symbol in batch:
            if symbol not in level0:
                print(f"No data returned for {symbol}")
                continue

            frame = raw[symbol].copy()

            if frame.empty:
                continue

            normalised = self._normalise(frame)

            if not normalised.empty:
                result[symbol] = normalised

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
