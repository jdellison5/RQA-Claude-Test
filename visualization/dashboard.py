"""
HTML dashboard generator for backtest results.

Produces a single self-contained HTML file with:
  - Equity curves (all strategies + buy-and-hold benchmark)
  - Drawdown chart (underwater equity per strategy)
  - Rolling 60-day Sharpe ratio
  - Rolling allocation (signal timeline: 100% invested vs 0%)
  - Performance stats comparison table
  - Trade log tables per strategy

Charts are rendered with Chart.js (loaded from CDN — internet required).
All data is embedded as JSON so the file works offline once opened.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper data builders
# ---------------------------------------------------------------------------

def _dates_to_str(index: pd.DatetimeIndex) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def _buy_hold_equity(data: pd.DataFrame, initial_capital: float) -> pd.Series:
    close = data["close"]
    shares = initial_capital / close.iloc[0]
    return (close * shares).rename("Buy & Hold")


def _rolling_sharpe(equity_curve: pd.Series, window: int = 60, periods_per_year: int = 252) -> pd.Series:
    returns = equity_curve.pct_change()
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)
    return sharpe.fillna(0)


def _drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return ((equity_curve - running_max) / running_max * 100)


def _signal_to_allocation(signal: pd.Series) -> pd.Series:
    """Convert 0/1 signal to allocation percentage (0% or 100%)."""
    return (signal * 100).astype(float)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS = [
    "#4C9BE8",  # blue   — strategy 1
    "#E8834C",  # orange — strategy 2
    "#4CE8A0",  # green  — strategy 3
    "#E84C7A",  # pink   — strategy 4
]
BENCHMARK_COLOR = "#9BA3B5"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_html_dashboard(
    results: list[tuple[str, object]],   # list of (label, BacktestResult)
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 10_000.0,
    output_path: str = "outputs/dashboard.html",
) -> str:
    """
    Build a self-contained HTML dashboard and write it to output_path.

    Args:
        results: list of (display_name, BacktestResult) pairs
        ticker:  stock ticker shown in header
        start:   backtest start date string
        end:     backtest end date string
        initial_capital: starting portfolio value
        output_path: where to write the HTML file

    Returns:
        The absolute path to the generated file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Use the first result's data for shared date index + benchmark
    first_data = results[0][1].data
    dates = _dates_to_str(first_data.index)
    bh = _buy_hold_equity(first_data, initial_capital)

    # -----------------------------------------------------------------------
    # Build all chart datasets as plain Python dicts → will be JSON-serialised
    # -----------------------------------------------------------------------

    # 1. Equity curves
    equity_datasets = []
    for i, (label, result) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        ec = result.equity_curve.reindex(first_data.index).ffill()
        equity_datasets.append({
            "label": label,
            "data": [round(v, 2) for v in ec.tolist()],
            "borderColor": color,
            "backgroundColor": color + "22",
            "borderWidth": 2,
            "pointRadius": 0,
            "fill": False,
            "tension": 0.1,
        })
    equity_datasets.append({
        "label": "Buy & Hold",
        "data": [round(v, 2) for v in bh.tolist()],
        "borderColor": BENCHMARK_COLOR,
        "backgroundColor": BENCHMARK_COLOR + "22",
        "borderWidth": 1.5,
        "borderDash": [6, 3],
        "pointRadius": 0,
        "fill": False,
        "tension": 0.1,
    })

    # 2. Drawdown
    drawdown_datasets = []
    for i, (label, result) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        ec = result.equity_curve.reindex(first_data.index).ffill()
        dd = _drawdown_series(ec)
        drawdown_datasets.append({
            "label": label,
            "data": [round(v, 2) for v in dd.tolist()],
            "borderColor": color,
            "backgroundColor": color + "44",
            "borderWidth": 1.5,
            "pointRadius": 0,
            "fill": True,
            "tension": 0.1,
        })

    # 3. Rolling Sharpe
    rolling_sharpe_datasets = []
    for i, (label, result) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        ec = result.equity_curve.reindex(first_data.index).ffill()
        rs = _rolling_sharpe(ec, window=60)
        rolling_sharpe_datasets.append({
            "label": label,
            "data": [round(v, 3) for v in rs.tolist()],
            "borderColor": color,
            "backgroundColor": "transparent",
            "borderWidth": 1.5,
            "pointRadius": 0,
            "fill": False,
            "tension": 0.1,
        })

    # 4. Rolling allocation (signal timeline)
    allocation_datasets = []
    for i, (label, result) in enumerate(results):
        color = COLORS[i % len(COLORS)]
        signal = result.data["signal"].reindex(first_data.index).fillna(0)
        alloc = _signal_to_allocation(signal)
        allocation_datasets.append({
            "label": label,
            "data": [round(v, 1) for v in alloc.tolist()],
            "borderColor": color,
            "backgroundColor": color + "55",
            "borderWidth": 1.5,
            "pointRadius": 0,
            "fill": True,
            "stepped": True,
        })

    # -----------------------------------------------------------------------
    # Performance stats table data
    # -----------------------------------------------------------------------

    def _fmt_pct(v):
        return f"{v:.2%}" if isinstance(v, float) else str(v)

    def _fmt_float(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    def _color_class(v, positive_good=True):
        if not isinstance(v, float) or v == 0:
            return "neutral"
        if positive_good:
            return "positive" if v > 0 else "negative"
        return "positive" if v < 0 else "negative"

    stat_rows = []
    metrics_list = [
        ("CAGR",                 "cagr",                       _fmt_pct,   True),
        ("Total Return",         "total_return",                _fmt_pct,   True),
        ("Sharpe Ratio",         "sharpe_ratio",                _fmt_float, True),
        ("Max Drawdown",         "max_drawdown",                _fmt_pct,   False),
        ("Max DD Duration (d)",  "max_drawdown_duration_days",  str,        False),
        ("Win Rate",             "win_rate",                    _fmt_pct,   True),
        ("Profit Factor",        "profit_factor",               _fmt_float, True),
        ("Total Trades",         "total_trades",                str,        True),
        ("Final Equity",         "final_equity",                lambda v: f"${v:,.2f}", True),
    ]

    for metric_label, key, fmt, pos_good in metrics_list:
        row = {"label": metric_label, "cells": []}
        for label, result in results:
            val = result.metrics.get(key, 0)
            raw = val if isinstance(val, (int, float)) else 0
            row["cells"].append({
                "text": fmt(val),
                "cls": _color_class(raw if isinstance(raw, float) else float(raw), pos_good),
            })
        stat_rows.append(row)

    # -----------------------------------------------------------------------
    # Trade log tables
    # -----------------------------------------------------------------------

    trade_tables = []
    for label, result in results:
        tl = result.trade_log
        if tl.empty:
            rows = []
        else:
            rows = []
            for _, t in tl.iterrows():
                pnl = t.get("pnl", 0) or 0
                rows.append({
                    "entry": str(t.get("entry_date", ""))[:10],
                    "exit":  str(t.get("exit_date",  ""))[:10],
                    "entry_price": f"${t.get('entry_price', 0):.2f}",
                    "exit_price":  f"${t.get('exit_price', 0):.2f}",
                    "pnl":   f"${pnl:+.2f}",
                    "pnl_cls": "positive" if pnl >= 0 else "negative",
                })
        trade_tables.append({"label": label, "rows": rows})

    # -----------------------------------------------------------------------
    # Serialize everything to JSON
    # -----------------------------------------------------------------------

    chart_data = json.dumps({
        "dates": dates,
        "equity": equity_datasets,
        "drawdown": drawdown_datasets,
        "rolling_sharpe": rolling_sharpe_datasets,
        "allocation": allocation_datasets,
    })
    stats_data   = json.dumps(stat_rows)
    trades_data  = json.dumps(trade_tables)
    strategy_labels = json.dumps([label for label, _ in results])

    # -----------------------------------------------------------------------
    # Render HTML
    # -----------------------------------------------------------------------

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Dashboard — {ticker}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f1117; color: #e0e0e0; min-height: 100vh; }}
  .header {{ background: #1a1d27; border-bottom: 1px solid #2d3148;
             padding: 20px 32px; display: flex; align-items: center;
             justify-content: space-between; }}
  .header h1 {{ font-size: 1.4rem; font-weight: 600; color: #fff; }}
  .header .meta {{ font-size: 0.8rem; color: #7a8099; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
  .card {{ background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px;
           padding: 20px; margin-bottom: 20px; }}
  .card h2 {{ font-size: 0.85rem; font-weight: 600; color: #7a8099;
              text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px; }}
  .chart-wrap {{ position: relative; height: 280px; }}
  .chart-wrap.tall {{ height: 340px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  thead th {{ color: #7a8099; font-weight: 600; font-size: 0.75rem;
              text-transform: uppercase; letter-spacing: 0.05em;
              padding: 8px 12px; border-bottom: 1px solid #2d3148; text-align: left; }}
  tbody td {{ padding: 9px 12px; border-bottom: 1px solid #1f2235; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: #1f2235; }}
  .metric-label {{ color: #a0a8c0; }}
  .positive {{ color: #4CE8A0; font-weight: 600; }}
  .negative {{ color: #E84C7A; font-weight: 600; }}
  .neutral  {{ color: #e0e0e0; }}
  .tab-row {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab {{ padding: 6px 14px; border-radius: 6px; cursor: pointer;
          font-size: 0.8rem; font-weight: 600; border: 1px solid #2d3148;
          background: transparent; color: #7a8099; transition: all 0.15s; }}
  .tab.active {{ background: #4C9BE8; color: #fff; border-color: #4C9BE8; }}
  .trade-panel {{ display: none; }}
  .trade-panel.active {{ display: block; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; margin-left: 6px; }}
  .badge-blue {{ background: #4C9BE822; color: #4C9BE8; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Backtest Dashboard &nbsp;<span style="color:#4C9BE8">{ticker}</span></h1>
    <div class="meta">{start} &rarr; {end} &nbsp;&bull;&nbsp; Generated {generated_at}</div>
  </div>
  <div class="meta" style="text-align:right">Initial capital: $10,000 &nbsp;&bull;&nbsp;
    {len(results)} strateg{"y" if len(results)==1 else "ies"}</div>
</div>

<div class="container">

  <!-- Equity Curves -->
  <div class="card">
    <h2>Equity Curves <span class="badge badge-blue">vs Buy &amp; Hold</span></h2>
    <div class="chart-wrap tall"><canvas id="equityChart"></canvas></div>
  </div>

  <div class="grid-2">
    <!-- Drawdown -->
    <div class="card">
      <h2>Drawdown (underwater equity)</h2>
      <div class="chart-wrap"><canvas id="drawdownChart"></canvas></div>
    </div>
    <!-- Rolling Sharpe -->
    <div class="card">
      <h2>Rolling 60-Day Sharpe Ratio</h2>
      <div class="chart-wrap"><canvas id="sharpeChart"></canvas></div>
    </div>
  </div>

  <!-- Rolling Allocation -->
  <div class="card">
    <h2>Rolling Allocation (% Invested in Market)</h2>
    <div class="chart-wrap"><canvas id="allocationChart"></canvas></div>
  </div>

  <!-- Stats Table -->
  <div class="card">
    <h2>Strategy Performance Comparison</h2>
    <table id="statsTable"></table>
  </div>

  <!-- Trade Logs -->
  <div class="card">
    <h2>Trade Log</h2>
    <div class="tab-row" id="tradeTabs"></div>
    <div id="tradePanels"></div>
  </div>

</div>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const CHART_DATA    = {chart_data};
const STATS_DATA    = {stats_data};
const TRADES_DATA   = {trades_data};
const STRATEGY_LABELS = {strategy_labels};

// ── Chart.js defaults ──────────────────────────────────────────────────────
Chart.defaults.color = "#7a8099";
Chart.defaults.borderColor = "#2d3148";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 11;

const sharedScales = {{
  x: {{
    type: "category",
    ticks: {{ maxTicksLimit: 8, maxRotation: 0 }},
  }},
  y: {{
    ticks: {{ maxTicksLimit: 6 }},
    grid: {{ color: "#2d314888" }},
  }},
}};

function makeChart(id, type, datasets, extraScales={{}}, extraOptions={{}}) {{
  return new Chart(document.getElementById(id), {{
    type,
    data: {{ labels: CHART_DATA.dates, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {{ mode: "index", intersect: false }},
      plugins: {{
        legend: {{ position: "top", labels: {{ boxWidth: 12, padding: 16 }} }},
        tooltip: {{ padding: 10, boxPadding: 4 }},
      }},
      scales: {{ ...sharedScales, ...extraScales }},
      ...extraOptions,
    }},
  }});
}}

// ── 1. Equity ──────────────────────────────────────────────────────────────
makeChart("equityChart", "line", CHART_DATA.equity, {{
  y: {{
    ticks: {{ callback: v => "$" + v.toLocaleString(), maxTicksLimit: 6 }},
    grid: {{ color: "#2d314888" }},
  }},
}});

// ── 2. Drawdown ────────────────────────────────────────────────────────────
makeChart("drawdownChart", "line", CHART_DATA.drawdown, {{
  y: {{
    ticks: {{ callback: v => v.toFixed(1) + "%", maxTicksLimit: 5 }},
    grid: {{ color: "#2d314888" }},
  }},
}});

// ── 3. Rolling Sharpe ──────────────────────────────────────────────────────
makeChart("sharpeChart", "line", CHART_DATA.rolling_sharpe, {{
  y: {{
    ticks: {{ maxTicksLimit: 5 }},
    grid: {{ color: "#2d314888" }},
  }},
}});

// ── 4. Allocation ──────────────────────────────────────────────────────────
makeChart("allocationChart", "line", CHART_DATA.allocation, {{
  y: {{
    min: 0, max: 110,
    ticks: {{ callback: v => v + "%", maxTicksLimit: 4 }},
    grid: {{ color: "#2d314888" }},
  }},
}});

// ── 5. Stats table ─────────────────────────────────────────────────────────
(function() {{
  const table = document.getElementById("statsTable");
  // Header
  let thead = "<thead><tr><th>Metric</th>";
  STRATEGY_LABELS.forEach(l => thead += `<th>${{l}}</th>`);
  thead += "</tr></thead><tbody>";
  // Rows
  let tbody = "";
  STATS_DATA.forEach(row => {{
    tbody += `<tr><td class="metric-label">${{row.label}}</td>`;
    row.cells.forEach(c => tbody += `<td class="${{c.cls}}">${{c.text}}</td>`);
    tbody += "</tr>";
  }});
  table.innerHTML = thead + tbody + "</tbody>";
}})();

// ── 6. Trade logs ──────────────────────────────────────────────────────────
(function() {{
  const tabRow   = document.getElementById("tradeTabs");
  const panels   = document.getElementById("tradePanels");

  TRADES_DATA.forEach((t, i) => {{
    // Tab button
    const btn = document.createElement("button");
    btn.className = "tab" + (i === 0 ? " active" : "");
    btn.textContent = t.label + " (" + t.rows.length + ")";
    btn.onclick = () => {{
      document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".trade-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("panel-" + i).classList.add("active");
    }};
    tabRow.appendChild(btn);

    // Panel
    const div = document.createElement("div");
    div.className = "trade-panel" + (i === 0 ? " active" : "");
    div.id = "panel-" + i;

    if (t.rows.length === 0) {{
      div.innerHTML = '<p style="color:#7a8099;padding:12px 0">No completed trades.</p>';
    }} else {{
      let html = `<table>
        <thead><tr>
          <th>Entry Date</th><th>Exit Date</th>
          <th>Entry $</th><th>Exit $</th><th>PnL</th>
        </tr></thead><tbody>`;
      t.rows.forEach(r => {{
        html += `<tr>
          <td>${{r.entry}}</td><td>${{r.exit}}</td>
          <td>${{r.entry_price}}</td><td>${{r.exit_price}}</td>
          <td class="${{r.pnl_cls}}">${{r.pnl}}</td>
        </tr>`;
      }});
      html += "</tbody></table>";
      div.innerHTML = html;
    }}
    panels.appendChild(div);
  }});
}})();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return os.path.abspath(output_path)
