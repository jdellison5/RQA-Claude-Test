# Technical Analysis Backtesting Framework

A Python backtesting framework built live during the University of Richmond guest lecture on Claude Code (April 1, 2025). Built from an empty repository in ~45 minutes using Claude Code.

## Project Structure

```
├── backtester/          # Core engine, portfolio tracking, data fetching, metrics
├── strategies/          # Abstract base + MA Crossover and RSI Mean Reversion
├── visualization/       # Matplotlib charts and console summary tables
├── tests/               # ~25 pytest tests using synthetic data (no network needed)
├── outputs/             # Generated charts (created at runtime)
├── .cache/              # Cached yfinance data (created at runtime)
└── run_demo.py          # Single entry-point script
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `yfinance`, `ta`, `pytest`

> **Note:** Use the `ta` library, not `pandas-ta`. `pandas-ta` does not support Python 3.11+.

## Running the Demo

```bash
# Run both strategies on AAPL (2020–2024), save charts to outputs/
python run_demo.py

# Custom ticker and date range
python run_demo.py --ticker MSFT --start 2021-01-01 --end 2025-01-01

# Include RSI oversold parameter sweep (25, 30, 35)
python run_demo.py --ticker AAPL --sweep-rsi
```

This will:
1. Fetch (or load from cache) OHLCV data for the specified ticker
2. Run Moving Average Crossover (fast=20, slow=50) and RSI Mean Reversion (period=14)
3. Print a metrics table to the console for each strategy
4. Save PNG charts to `outputs/`

## Running Tests

```bash
pytest -v
```

All ~25 tests use synthetic price data and run in under 5 seconds. No network access required.

## Interpreting the Output

**Console metrics table:**

| Metric | What it means |
|--------|--------------|
| CAGR | Compound annual growth rate of the strategy |
| Sharpe Ratio | Risk-adjusted return (>1 is good, >2 is excellent) |
| Max Drawdown | Largest peak-to-trough equity decline (more negative = worse) |
| Win Rate | Fraction of trades that were profitable |
| Profit Factor | Gross profit / gross loss (>1 is profitable) |
| Total Trades | Number of complete round-trip trades |

Compare these to a **buy-and-hold benchmark** — visible on the equity curve chart as a dashed gray line.

## Key Concepts Demonstrated

### Look-Ahead Bias Prevention
Every strategy uses `.shift(1)` on signals before returning them:

```python
# WRONG — uses today's close to generate AND act on a signal
signal = np.where(sma_fast > sma_slow, 1, 0)

# CORRECT — signal is generated from today's close but acted on tomorrow's open
signal = pd.Series(np.where(sma_fast > sma_slow, 1, 0)).shift(1)
```

This is one of the most common mistakes in backtesting. The test suite explicitly checks for it via `test_no_lookahead_bias`.

### Strategy Interface
All strategies inherit from `BaseStrategy` and implement one method:

```python
def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
    # Return data enriched with 'signal' column (1=long, 0=flat)
```

The engine calls this once, then replays execution bar-by-bar. Signal generation and execution are fully decoupled.

## Extending the Framework

### Adding a New Strategy

1. Create `strategies/your_strategy.py`:

```python
from strategies.base import BaseStrategy
import pandas as pd

class YourStrategy(BaseStrategy):
    def __init__(self, param: int = 10) -> None:
        super().__init__(name=f"YourStrategy({param})")
        self.param = param

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # ... compute indicators ...
        # ALWAYS shift by 1 to prevent look-ahead bias
        df["signal"] = your_raw_signal.shift(1).fillna(0).astype(int)
        return df
```

2. Run it:

```python
from backtester.engine import Backtester
from backtester.data import load_or_fetch
from strategies.your_strategy import YourStrategy

data = load_or_fetch("AAPL", "2020-01-01", "2024-01-01")
bt = Backtester(strategy=YourStrategy(param=20), initial_capital=10_000)
result = bt.run(data)
print(result.metrics)
```

### Adding a New Metric

Add a function to `backtester/metrics.py` and include it in `summary_stats()`:

```python
def calmar_ratio(equity_curve: pd.Series) -> float:
    """CAGR divided by absolute max drawdown."""
    dd = abs(max_drawdown(equity_curve))
    if dd == 0:
        return float("inf")
    return cagr(equity_curve) / dd
```

### MACD Strategy Example

```bash
# Ask Claude Code:
"Add a MACDCrossover strategy to strategies/macd_crossover.py using
ta.trend.MACD. Follow the same interface as MovingAverageCrossover."
```
