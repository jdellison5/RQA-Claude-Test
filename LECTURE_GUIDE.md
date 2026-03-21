# Claude Code Live Demo Guide
### University of Richmond — Technical Analysis Class

> **Purpose:** Step-by-step script for rebuilding this backtesting framework live in front of a class, starting from an empty repository. Each prompt is copy-paste ready. Total runtime: ~50 minutes.

---

## Before Class

- [ ] Clone the repo and confirm Python 3.11 is available
- [ ] Run `pip install pandas numpy matplotlib yfinance ta pytest` once so packages are cached
- [ ] Run `python generate_dashboard.py` once to populate `.cache/SPY_*.csv` (protects against network issues during the lecture)
- [ ] Have this file open in a second window so you can paste prompts quickly
- [ ] Open the GitHub Actions tab in a browser tab so you can show CI results live

---

## The Narrative

> *"I'm going to show you how to build a fully tested, documented technical analysis backtesting framework from an empty repository — using Claude Code. We'll do it live. The repo starts empty. By the end, we'll have two strategies, an interactive dashboard, and a CI pipeline running tests in the cloud."*

---

## Act 1 — Scaffold the Project (10 min)

**What you're showing:** Claude Code can generate an entire professional project structure from a plain-English description in seconds.

### Prompt 1 — Project scaffold

```
Create a Python backtesting framework project structure. I need:
- A `backtester` package with engine.py, portfolio.py, data.py, and metrics.py
- A `strategies` package with a base class and two strategies:
  Moving Average Crossover and RSI Mean Reversion
- A `visualization` package with charts.py
- A `tests` directory with pytest fixtures
- A requirements.txt with pandas, numpy, matplotlib, yfinance, ta, and pytest
- A run_demo.py entry point

Use dataclasses where appropriate. Add type hints and docstrings throughout.
```

**Teaching points:**
- Notice it created all `__init__.py` files automatically
- Type hints throughout — this is what professional code looks like
- The abstract base class in `strategies/base.py` enforces a contract — every strategy must implement `generate_signals()`

---

## Act 2 — Implement the Core Engine (15 min)

**What you're showing:** Claude Code understands financial concepts, not just syntax.

### Prompt 2 — Portfolio

```
Implement the Portfolio class in backtester/portfolio.py.
It should track a single long position at a time using all available cash,
calculate P&L when a position closes, and maintain a daily equity curve.
Use a dataclass for Trade records.
```

### Prompt 3 — Engine

```
Implement the Backtester engine in backtester/engine.py.
It should call strategy.generate_signals() once up front, then iterate
bar by bar executing trades. Return a BacktestResult dataclass.

Important: prevent look-ahead bias — we should only act on signals from
the previous bar, not the current bar's close price.
```

**🔑 Teaching moment — look-ahead bias:**
> *"This is one of the most common mistakes in backtesting. You can't trade on a signal computed from today's closing price — the market is still open. Claude Code used `.shift(1)` to lag the signal by one bar. We'll test this explicitly later."*

Draw on the board:
```
Bar t closes → signal computed → signal.shift(1) → trade executes at Bar t+1 open
```

### Prompt 4 — Metrics

```
Implement backtester/metrics.py with these functions:
sharpe_ratio, max_drawdown, max_drawdown_duration, cagr, win_rate, profit_factor.
Include the full formula as a docstring for each.
```

**Teaching moment:** Walk through the Sharpe ratio formula — students know it from portfolio theory. Connect the formula to what they're about to see in the output.

---

## Act 3 — Strategies + Iteration (15 min)

**What you're showing:** The iterative workflow. Ask → run → ask again.

### Prompt 5 — MA Crossover strategy

```
Implement the MovingAverageCrossover strategy in strategies/ma_crossover.py.
Fast window default 20 days, slow window default 50 days.
Add SMA columns to the dataframe and generate a buy signal (1) when the
fast SMA crosses above the slow SMA, and a sell signal (0) when it crosses below.
Remember to shift the signal by 1 bar to prevent look-ahead bias.
```

### Run it

```bash
python run_demo.py --ticker SPY --start 2020-01-01 --end 2024-01-01
```

Show the console metrics table. Discuss each metric:
- **CAGR** — annualized growth rate
- **Sharpe** — return per unit of risk (>1 is good, >2 is excellent)
- **Max Drawdown** — worst peak-to-trough loss
- **Win Rate** — fraction of trades that were profitable

### Prompt 6 — Add benchmark to chart

```
Modify visualization/charts.py to add a buy-and-hold benchmark line
to the equity curve panel. Show the benchmark as a dashed gray line.
Also add benchmark CAGR and benchmark Sharpe to the metrics table.
```

**Teaching moment:**
> *"I didn't describe how to calculate buy-and-hold returns. Claude Code already knows. And it only modified the files that needed to change — it didn't touch the engine or the strategies."*

### Prompt 7 — RSI strategy

```
Implement the RSIMeanReversion strategy in strategies/rsi_mean_reversion.py
using the `ta` library's RSIIndicator (not pandas-ta — not compatible with Python 3.11).
Generate a buy signal when RSI crosses up through 30,
and a sell signal when RSI crosses down through 70.
Shift the signal by 1 bar to prevent look-ahead bias.
```

**Ask the class:** *"What do you expect to happen if we change the oversold threshold from 30 to 25? More trades or fewer? Better or worse Sharpe?"*

### Prompt 8 — Parameter sweep

```
Add a --sweep-rsi flag to run_demo.py that tests RSI oversold levels
of [25, 30, 35] and prints a comparison table showing CAGR, Sharpe,
max drawdown, and trade count for each.
```

Run it. Let the class interpret the results.

---

## Act 4 — Tests + Bug Demo (10 min)

**What you're showing:** Claude Code writes tests that catch real financial logic bugs.

### Prompt 9 — Full test suite

```
Write a comprehensive pytest test suite for this project in the tests/ directory.
Use synthetic price data (no network calls) generated with numpy and np.random.seed(42).
Include tests for:
- Portfolio P&L calculations with known expected values
- The engine's look-ahead bias prevention
- Metrics with manually verifiable expected values (e.g. an equity curve that
  doubles in 252 days should have CAGR ≈ 100%)
- Strategy signals always being valid (only 0 or 1)
```

Run the tests:

```bash
pytest tests/ -v
```

Watch ~46 green tests appear. Note: "Under 0.5 seconds. No network calls. Fully deterministic."

### 🎭 The bug demo (most dramatic moment)

**Step 1** — Intentionally break look-ahead bias:

Open `strategies/ma_crossover.py` and remove the `.shift(1)`:
```python
# Change this:
df["signal"] = pd.Series(raw_signal, index=df.index).shift(1).fillna(0).astype(int)

# To this (broken):
df["signal"] = pd.Series(raw_signal, index=df.index).fillna(0).astype(int)
```

**Step 2** — Run tests:
```bash
pytest tests/ -v
```

The `test_ma_crossover_no_lookahead_bias` test fails with a clear message.

**Step 3** — Fix it:
```
Fix the look-ahead bias in strategies/ma_crossover.py.
```

Tests pass again.

**Teaching moment:**
> *"In production, look-ahead bias makes your backtest look fantastic. You'd think you had a great strategy. You'd deploy it. You'd lose money — because the live system can't see tomorrow's close. The test caught it. This is why we write tests."*

---

## Act 5 — Dashboard + CI (5 min)

### Generate the HTML dashboard

```bash
python generate_dashboard.py --ticker SPY --start 2020-01-01 --end 2024-01-01
open outputs/dashboard.html
```

Walk through the dashboard sections:
1. **Equity curves** — all strategies vs buy-and-hold
2. **Drawdown** — how far underwater each strategy got
3. **Rolling Sharpe** — does the edge persist over time, or was it luck?
4. **Rolling allocation** — when was each strategy invested vs in cash?
5. **Correlation matrix** — are the strategies actually different from each other?
6. **Trade log** — every single trade, with P&L

**On the correlation matrix:**
> *"If two strategies are highly correlated, running both doesn't reduce your risk much — you're essentially doubling down. A low or negative correlation between strategies is where real portfolio diversification comes from."*

### Show CI

Open the GitHub Actions tab. Show a green run. Explain:
> *"Every time we push code, GitHub spins up a fresh server, installs all our dependencies, fetches real SPY data from Yahoo Finance, and runs all 46 tests automatically. If anything breaks, we get an email before it reaches production."*

---

## Closing (2 min)

> *"In under an hour, we went from an empty repository to:*
> - *A tested, documented backtesting framework*
> - *Two technical strategies with look-ahead bias protection*
> - *An interactive HTML dashboard with correlation analysis*
> - *A CI pipeline running against real market data in the cloud*
>
> *Claude Code handled the boilerplate, remembered the patterns, and enforced consistency. Your job — as a quant, as any engineer — is to ask the right questions and understand the output. That judgment is what Claude Code cannot replace."*

---

## If Students Ask...

| Question | Answer |
|---|---|
| "Why `ta` and not `pandas-ta`?" | `pandas-ta` doesn't support Python 3.11+. `ta` is pure Python, always installable. |
| "Can we add MACD?" | "Great idea — prompt: *Add a MACDCrossover strategy using `ta.trend.MACD`, same interface as MovingAverageCrossover.*" Takes 2 min. |
| "Why not use TA-Lib?" | TA-Lib requires a compiled C binary. Painful to install. `ta` just works with pip. |
| "Can this run on live data?" | Yes — replace `load_or_fetch` with a streaming data source. The engine doesn't care where the DataFrame comes from. |
| "What's look-ahead bias?" | Using future information to make a past decision. Like betting on a horse race after knowing the results. |
| "Why `iterrows()` and not vectorized?" | Readability and correctness first. For a lecture, you want students to see each bar processed one at a time. Production would vectorize. |

---

## Key Files for Reference

| File | What it does |
|---|---|
| `backtester/engine.py` | Bar-by-bar execution loop |
| `backtester/portfolio.py` | Position tracking and P&L math |
| `backtester/metrics.py` | Sharpe, drawdown, CAGR, win rate, profit factor |
| `backtester/data.py` | yfinance fetch + local CSV cache |
| `strategies/base.py` | Abstract interface all strategies must implement |
| `strategies/ma_crossover.py` | SMA Golden/Death Cross |
| `strategies/rsi_mean_reversion.py` | RSI oversold/overbought using `ta` library |
| `visualization/dashboard.py` | Generates self-contained HTML dashboard |
| `tests/conftest.py` | Shared fixtures (SPY data + synthetic fallback) |
| `run_demo.py` | CLI entry point for console output + PNG charts |
| `generate_dashboard.py` | CLI entry point for HTML dashboard |
| `.github/workflows/ci.yml` | GitHub Actions CI — runs pytest on every push |
