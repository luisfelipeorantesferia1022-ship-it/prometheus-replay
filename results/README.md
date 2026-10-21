# Prometheus 2025 replay: recorded data

Full-year 2025 walk-forward replay of the Prometheus workflow (weekend LSTM sweep
-> stage 2 intraday backtest -> sim 1 -> sim 2 -> promotion -> V3 agent), run on
GitHub Actions from the `prometheus-replay` repo. Raw Sharadar and Databento
market data are not included (licensed, and kept only in the GitHub Actions cache).

Start with `report/replay_2025_tables.html` (open in a browser).

| Folder / file | What it holds |
|---|---|
| `report/replay_2025_tables.html` | Tables for the LSTM real, 1-trade min book (9:46 entry + 10:01 spread re-check): trades, held to expiry and best prices, the factor study (funnel, option, alpha/beta, peak breakdown, correlations) |
| `weekly/<as-of date>/` | Shared weekly funnel: `lstm_accuracy_results.csv` (stage 1, every S&P 500 member), `lstm_candidates.csv`, `stage1.json`, `intraday_backtest_results.csv` (stage 2), `stage2_gate.csv` (200% gate), `sim1_*` (sim 1) |
| `variants/sim_min1/`, `variants/sim_min3/` | 1-trade and 3-trade minimum variants: `whitelist_timeline.csv` (whitelist each week) and `weekly/<as-of>/` with `sim2_*` and `promotion_table.csv` |
| `agent/lstm_forecasts.csv` | LSTM forecast the agent saw each morning per whitelisted ticker |
| `agent/<variant>/<book>/` | One agent book each: `trades.csv`, `daily_equity.csv`, `entry_log.csv` (every morning's decision, skips and spread re-checks), `summary.json` |
| `analysis/hold_<variant>_<book>.csv` | Per trade: held-to-expiry P&L, best bid reached and when |
| `analysis/trade_factors.csv` | 23 distinct trades x ~80 measures (funnel, morning forecast, greeks, stock beta/alpha, option beta, peak breakdown) |
| `analysis/trade_factor_correlations.csv` | Each measure's Spearman correlation with the option peak |
| `trace/lineage.json` | Every agent trade traced back through promotion, stage 1, stage 2, sim 1, sim 2 and the entry morning |
| `databento_ledger.json` | Databento spend ($4.67 of the $40 cap) |
| `data_report.json` | Point-in-time data checks |

Book names: `lstm_copy` (what the sandbox agent would book: 15-min delayed quotes,
mid fills), `lstm_real` (real prices: ask in, bid out), `spy_rsi_real` (control,
no LSTM). Suffix `_0946` = entries wait until 9:46 ET; `_r1001` = 10:01 re-check
for tickers rejected only on the 20% spread rule.
