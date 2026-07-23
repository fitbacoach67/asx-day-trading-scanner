# ASX Short-Term Trading Scanner

A research-oriented Streamlit application that:

- scans a configurable universe of liquid ASX ordinary shares;
- detects unusual relative volume and short-term momentum;
- applies deterministic technical rules;
- ranks moderate/high-confidence long setups;
- stores qualifying shares in a SQLite watchlist;
- calculates entry, stop-loss, first target and second target;
- creates Yahoo Finance and TradingView chart links; and
- optionally uses Groq to turn the numeric evidence into a concise analysis summary.

> **Important:** This is not financial advice and it does not place orders. The default
> `yfinance` feed is unofficial and may be delayed, incomplete or unsuitable for live
> trading. Validate all signals against a licensed real-time ASX feed and your broker.

## Signal design

The scanner does not ask the LLM to choose prices or manufacture signals. Python calculates
all indicators and trade levels first. Groq receives only those computed facts and produces
a constrained explanation.

The score includes:

- 20-day median dollar turnover liquidity;
- relative volume versus the prior 20 bars;
- EMA 9/20 trend;
- RSI(14);
- MACD histogram;
- price position versus VWAP;
- 20-bar breakout proximity;
- ATR-normalised momentum; and
- reward-to-risk feasibility.

A candidate reaches:

- **High confidence:** score >= 78
- **Moderate confidence:** score >= 65
- **Low confidence:** score < 65

Only moderate/high candidates are automatically added to the watchlist.

## Local setup

```bash
git clone https://github.com/YOUR-USER/asx-day-trading-scanner.git
cd asx-day-trading-scanner

python -m venv .venv
source .venv/bin/activate              # macOS/Linux
# .venv\Scripts\activate               # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env
```

Add your Groq key to `.env`:

```dotenv
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Run:

```bash
streamlit run app.py
```

## GitHub setup

1. Create a new empty GitHub repository, for example `asx-day-trading-scanner`.
2. In this project folder run:

```bash
git init
git add .
git commit -m "Initial ASX scanner"
git branch -M main
git remote add origin https://github.com/YOUR-USER/asx-day-trading-scanner.git
git push -u origin main
```

3. Never commit `.env`. It is already excluded in `.gitignore`.
4. In GitHub, enable Dependabot and secret scanning if available.

## Streamlit Community Cloud deployment

1. Push the repository to GitHub.
2. In Streamlit Community Cloud select the repository and `app.py`.
3. Add secrets in the app settings:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

The app checks both environment variables and Streamlit secrets.

SQLite watchlists on Community Cloud are ephemeral. For durable multi-user deployment,
replace `watchlist.py` with PostgreSQL, Supabase or another managed database.

## Universe management

Edit `config/asx_liquid.csv`. The included list is a practical starter universe, not a
guaranteed current index constituent list. The scanner also applies a minimum median daily
dollar-turnover filter.

ASX Yahoo symbols use the `.AX` suffix, such as `BHP.AX`.

## Data modes

- **Daily:** more reliable for end-of-day swing/next-session setups.
- **60m / 30m / 15m:** experimental intraday views subject to data-source limits.

For genuine day trading, implement the `MarketDataProvider` interface in
`src/data_provider.py` using your licensed feed, then switch the provider in `app.py`.

## Tests

```bash
pytest -q
```

## Risk controls to add before real use

- real-time quote age checks;
- bid/ask spread and market-depth filters;
- trading halt and announcement checks;
- scheduled results/corporate-action exclusion;
- maximum position risk and portfolio heat;
- slippage and brokerage;
- walk-forward backtesting;
- paper-trading reconciliation; and
- broker-side hard stops.
