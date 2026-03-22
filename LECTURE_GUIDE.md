# Claude Code Live Demo Guide
### University of Richmond — Technical Analysis Class

> **Purpose:** A step-by-step script for building a stock backtesting tool live in front of the class, starting from nothing. Each prompt is copy-paste ready and written in plain English — no programming experience needed. Total runtime: ~50 minutes.

---

## Before Class

- [ ] Clone the repo and make sure Python is installed
- [ ] Run `pip install pandas numpy matplotlib yfinance ta pytest` once so packages are cached
- [ ] Run `python generate_dashboard.py` once to save a local copy of SPY data (protects against network issues during the lecture)
- [ ] Have this file open in a second window so you can paste prompts quickly
- [ ] Open the GitHub Actions tab in a browser tab so you can show CI results live

---

## The Narrative

> *"I'm going to show you how to build a stock trading strategy backtester from scratch — using plain English. No prior coding experience required. I'll just describe what I want, and Claude Code will build it. By the end, we'll have two real strategies, an interactive dashboard, and automated tests running in the cloud."*

---

## Act 1 — Build the Foundation (10 min)

**What you're showing:** You can describe a complex project in plain English and Claude Code will create the entire structure instantly.

### Prompt 1 — Set up the project

```
I want to build a stock trading backtester in Python. It should be able to:
- Download historical stock price data from the internet
- Run trading strategies on that data to simulate how they would have performed
- Track a portfolio over time and calculate performance metrics like returns, risk, and win rate
- Display the results visually

Please set up the full project structure for me, including everything needed to run tests.
```

**Teaching points:**
- Claude Code created every file and folder automatically — you didn't name a single one
- Notice it also created a `tests` folder — it's already thinking about quality
- Ask the class: *"What would this have taken a solo developer to set up? A day? A week?"*

---

## Act 2 — Build the Core Logic (15 min)

**What you're showing:** Claude Code understands finance, not just code.

### Prompt 2 — Track trades and a portfolio

```
Now build out the part that tracks our trades and portfolio value over time.
It should be able to:
- Start with a set amount of cash
- Buy shares with all available cash when we get a buy signal
- Sell all shares when we get a sell signal
- Record every trade and calculate profit or loss on each one
- Track the total value of our portfolio each day
```

### Prompt 3 — Build the backtesting engine

```
Build the backtesting engine — the part that actually runs a strategy against historical data.
It should go through the price data day by day and decide whether to buy or sell based on the strategy's signals.

One important rule: we should only be able to act on information we would have actually known at the time.
For example, if a signal triggers at the end of Monday, we shouldn't be able to trade until Tuesday —
we can't trade on a signal we didn't have yet.
```

**🔑 Teaching moment — look-ahead bias:**
> *"This is one of the most common mistakes in backtesting. Imagine betting on a horse race after you already know who won — of course you'd look like a genius. If your backtest lets tomorrow's data influence today's trade, the results look great, but the strategy is worthless in real life. Claude Code prevented this automatically."*

Draw on the board:
```
Monday closes → signal calculated → can only trade on Tuesday
```

### Prompt 4 — Calculate performance metrics

```
Add the ability to calculate standard performance metrics for any strategy we test:
- Annualized return (CAGR)
- Sharpe ratio (return relative to risk)
- Maximum drawdown (worst loss from a peak)
- Win rate (what percentage of trades were profitable)
- Profit factor (total gains divided by total losses)

Please include a brief plain-English explanation of what each metric means in the code itself.
```

**Teaching moment:** Walk through Sharpe ratio — students know it from portfolio theory. Connect the formula to what they're about to see in the output.

---

## Act 3 — Build the Strategies (15 min)

**What you're showing:** The iterative workflow. Ask → run → refine.

### Prompt 5 — Moving Average Crossover strategy

```
Build a Moving Average Crossover strategy.
The idea: calculate a short-term average (20 days) and a long-term average (50 days) of the stock's price.
Buy when the short-term average crosses above the long-term average — that's a "golden cross."
Sell when it crosses back below — that's a "death cross."

Make sure the strategy can't see future data when generating signals.
```

### Run it

```bash
python run_demo.py --ticker SPY --start 2020-01-01 --end 2024-01-01
```

Show the console metrics table. Discuss each metric:
- **CAGR** — if you'd invested on day one, what annual return did you earn?
- **Sharpe** — how much return did you get per unit of risk? (>1 is solid, >2 is great)
- **Max Drawdown** — at its worst, how far did the portfolio fall from its peak?
- **Win Rate** — what fraction of trades made money?

### Prompt 6 — Add a benchmark to compare against

```
Update the performance chart to include a "buy and hold" benchmark line —
what would have happened if we just bought the stock on day one and held it the whole time.
Show it as a dashed gray line so it's easy to distinguish from our strategy.
Also add the buy-and-hold return and risk metrics to the summary table so we can compare directly.
```

**Teaching moment:**
> *"I didn't explain how to calculate buy-and-hold returns. Claude Code already knew. And notice it only changed the chart code — it didn't touch the strategy or the engine."*

### Prompt 7 — RSI Mean Reversion strategy

```
Build a second strategy based on RSI (Relative Strength Index).
The idea: when a stock has been beaten down a lot (RSI below 30), it's oversold — buy it.
When it's been overbought (RSI above 70), sell it.

Again, make sure the strategy can only use information available at the time of the trade.
```

**Ask the class:** *"What do you expect to happen if we change the 'oversold' level from 30 to 25? Will we trade more or less often? Will the returns be better or worse?"*

### Prompt 8 — Test different RSI settings

```
Add an option to automatically test different RSI oversold thresholds — try 25, 30, and 35.
Show a comparison table with the return, Sharpe ratio, max drawdown, and number of trades for each setting.
```

Run it. Let the class interpret the results before you comment.

---

## Act 4 — Tests + Bug Demo (10 min)

**What you're showing:** Automated tests catch real financial logic errors before they cost money.

### Prompt 9 — Write a test suite

```
Write a full set of automated tests for everything we've built.
The tests should:
- Use fake, made-up price data so they don't need an internet connection
- Check that the portfolio calculates profits and losses correctly
- Verify that our strategies can't accidentally use future data
- Confirm that all the performance metrics produce correct results with known inputs
- Make sure strategy signals are always valid
```

Run the tests:

```bash
pytest tests/ -v
```

Watch the green checkmarks appear. Note: *"Under half a second. No internet required. Same result every time."*

### The bug demo — most dramatic moment

**Step 1** — Intentionally break the look-ahead bias protection:

Open `strategies/ma_crossover.py` and remove the `.shift(1)`:
```python
# Change this (correct):
df["signal"] = pd.Series(raw_signal, index=df.index).shift(1).fillna(0).astype(int)

# To this (broken — strategy can now "see the future"):
df["signal"] = pd.Series(raw_signal, index=df.index).fillna(0).astype(int)
```

**Step 2** — Run the tests:
```bash
pytest tests/ -v
```

One test fails with a clear message pointing directly to the problem.

**Step 3** — Fix it with Claude Code:
```
The look-ahead bias test is failing in the moving average strategy. Please fix it.
```

Tests go green again.

**Teaching moment:**
> *"Without that test, this bug could have gone undetected. The backtest results would have looked great — because the strategy was secretly cheating. You'd deploy it with real money, and you'd lose — because in real life, you can't trade on tomorrow's close. The test caught it instantly."*

---

## Act 5 — Dashboard + CI (5 min)

### Generate the interactive dashboard

```bash
python generate_dashboard.py --ticker SPY --start 2020-01-01 --end 2024-01-01
open outputs/dashboard.html
```

Walk through each section:
1. **Equity curves** — all strategies vs buy-and-hold, over time
2. **Drawdown** — how far underwater each strategy got at its worst
3. **Rolling Sharpe** — is the edge consistent, or did it just get lucky in one period?
4. **Rolling allocation** — when was each strategy in the market vs sitting in cash?
5. **Correlation matrix** — are the two strategies actually doing different things?
6. **Trade log** — every single trade, with entry, exit, and profit/loss

**On the correlation matrix:**
> *"If two strategies move together — both up on the same days, both down on the same days — then running both doesn't actually reduce your risk. Real diversification comes from strategies that don't move in lockstep."*

### Show CI

Open the GitHub Actions tab. Show a green run. Explain:
> *"Every time we push code, GitHub automatically spins up a fresh computer in the cloud, installs everything, downloads real SPY data, and runs all the tests. If anything breaks, we get notified before it ever reaches production. We didn't write any of that pipeline — we asked Claude Code to set it up."*

---

## Closing (2 min)

> *"In under an hour, we went from nothing to:*
> - *A working backtesting framework that handles real market data*
> - *Two technical strategies with built-in protection against cheating*
> - *An interactive dashboard with correlation analysis*
> - *Automated tests running in the cloud on every change*
>
> *The only thing we wrote was plain English. Claude Code handled everything else.*
>
> *Your value — as a quant, as any analyst — isn't writing the code. It's knowing enough to ask the right questions, and enough to recognize when the answer is wrong. That judgment is what Claude Code cannot replace."*

---

## If Students Ask...

| Question | Answer |
|---|---|
| "Can we add a MACD strategy?" | Absolutely — try: *"Add a MACD crossover strategy. Buy when the MACD line crosses above the signal line. Sell when it crosses below."* Takes about 2 minutes. |
| "What's look-ahead bias in plain English?" | Using future information to make a past decision. Like betting on a horse race after knowing the results — of course you'd look smart. |
| "Could this work with live data instead of historical?" | Yes. The strategy and engine don't care where the data comes from. You'd swap out the data source and plug in a real brokerage API. |
| "How is this different from just using Excel?" | Excel can do simple backtests, but it can't run automated tests, version control your logic, or scale to thousands of strategies. This approach is how it's actually done at hedge funds. |
| "Why does the RSI strategy trade so rarely?" | Mean reversion strategies wait for extreme conditions. Long stretches with no trades are normal — the strategy is patient by design. |
| "Could someone actually use this to make money?" | The framework is real. The strategies are simple examples. Real quant strategies involve much more rigorous testing, transaction cost modeling, and risk management. |

---

## Key Sections of the Codebase

| What it does | Where it lives |
|---|---|
| Downloads and caches stock price data | `backtester/data.py` |
| Goes through data day by day and executes trades | `backtester/engine.py` |
| Tracks cash, shares, and portfolio value | `backtester/portfolio.py` |
| Calculates Sharpe, drawdown, CAGR, win rate | `backtester/metrics.py` |
| Moving Average Crossover logic | `strategies/ma_crossover.py` |
| RSI Mean Reversion logic | `strategies/rsi_mean_reversion.py` |
| Generates the HTML dashboard | `visualization/dashboard.py` |
| Automated test suite | `tests/` |
| Runs tests automatically in the cloud | `.github/workflows/ci.yml` |
| Command-line entry point for console output | `run_demo.py` |
| Command-line entry point for the dashboard | `generate_dashboard.py` |
