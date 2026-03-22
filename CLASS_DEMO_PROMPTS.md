# Live Class Demo — Prompt Sheet

---

## Step 1 — Build the foundation

```
I want to build a stock trading strategy backtester in Python from scratch.
It should be able to download historical price data, run trading strategies
against it, track a simulated portfolio over time, and calculate performance
metrics like annual return, Sharpe ratio, max drawdown, and win rate.
Please set up the full project for me.
```

---

## Step 2 — Build Strategy 1

> **Ask the class:** *What should the buy and sell rules be?*
> Fill in the blanks below based on their answers.

```
Build a trading strategy with these rules:
- Buy when: [CLASS DECIDES]
- Sell when: [CLASS DECIDES]

Make sure the strategy can never use information it wouldn't have had at the time —
it can only act on yesterday's signal, not today's.
```

---

## Step 3 — Build Strategy 2

> **Ask the class:** *What's a different approach we could test?*

```
Build a second strategy with different rules:
- Buy when: [CLASS DECIDES]
- Sell when: [CLASS DECIDES]

Same rule — no peeking at future data.
```

---

## Step 4 — Write automated tests

```
Write a full set of automated tests for everything we've built.
The tests should use made-up price data so they don't need an internet connection,
and they should verify that our strategies can never accidentally use future data.
```

---

## Step 5 — Build the dashboard

```
Build an interactive HTML dashboard that shows:
- How each strategy performed versus just buying and holding
- How far each strategy fell from its peak at its worst
- Whether the two strategies tend to move together or independently
- A log of every trade with the profit or loss on each one

Save it as a standalone HTML file that anyone can open in a browser.
```

---

## Step 6 — Set up GitHub Actions and GitHub Pages

```
Set up a GitHub Actions workflow that automatically runs all our tests
whenever we push new code. Also set up GitHub Pages so that the dashboard
is rebuilt and published to a public URL on every push.
```

---

## Step 7 — Run it

```bash
python generate_dashboard.py
```

Then push to GitHub and show the class the live URL.
