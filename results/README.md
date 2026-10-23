# Prometheus 2025 replay: recorded data

Full-year 2025 walk-forward replay of the Prometheus workflow (weekend LSTM sweep
-> stage 2 intraday backtest -> sim 1 -> sim 2 -> promotion -> V3 agent), run on
GitHub Actions from the `prometheus-replay` repo. Raw Sharadar and Databento
market data are not included (licensed, and kept only in the GitHub Actions cache).

Start with the pages in `report/` (open in a browser): `replay_2025_tables.html`, `calm_test_tables.html`, `winner_search_tables.html`, `sell_strategy.html`, `oos_spread_test.html`.

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
| `report/calm_test_tables.html` | Calm-stock test: do calmer, lower-error stocks give bigger option peaks? (real OPRA quotes, 1,518 options, 52 weeks) |
| `analysis/calm_test.csv` | One row per option in that test: stock, call/put, contract, price paid (ask at 9:46), implied vol, leverage, LSTM error, volatility, beta, peak, peak within 5 days, value at the Thursday exit |
| `analysis/calm_summary.json` | The test's correlations, week-by-week counts and fifths tables |
| `report/winner_search_tables.html` | Winner/loser search: can anything known at 9:46 pick winners or weed out losers? (39 features, checked in both halves of 2025) |
| `analysis/calm_features.csv` | The calm-test options plus entry-time features: gap, early move, momentum, RSI, 52-week range, volume, earnings timing, SPY regime, option cost |
| `analysis/winner_search.json` | The search results: per-feature correlations by half, models, call/put side rules |
| `report/sell_strategy.html` | Sell-strategy lab: a chart for every agent trade, the common price patterns, ~2,600 sell rules compared on both halves of 2025 |
| `analysis/exit_paths.parquet` | 5-minute bid/mid paths for the 1,518 test options (entry to the Thursday exit) |
| `analysis/agent_trade_paths.parquet` | 1-minute bid/ask/stock paths for the agent's 23 trades, entry to expiry |
| `analysis/exit_rules.csv` | Every sell rule's results, by half of the year |
| `analysis/exit_v3_check.csv`, `analysis/exit_lab.json` | Live V3 rules at 1-minute precision per option; pattern statistics |
| `analysis/spread_check.csv` | The 1,518 test options with entry spread and the return under each sell rule (V3 exact, V3 + day-2, day-2, hold, +50% take-profit) |
| `report/oos_spread_test.html` | Out-of-sample test of the tight-spread rule on 2024 and Jan-Sep 2026 (pre-registered): verdict, the four checks, other sell rules, splits, month by month |
| `oos/PREREGISTRATION.md`, `oos/README.md` | The test's locked rule and pass criteria (written before the data), and its outcome and run notes |
| `oos/oos_trades.csv`, `oos/verdict.json`, `oos/universe.csv` | Every contract measured (entry quote, spread, delta, return under each sell rule), the scored verdict, and the weekly point-in-time sample |
| `analysis/oracle_test.txt` | Oracle test (replay/oracle_test.py): if a model knew the stock's direction perfectly over a given horizon, what V3 would have earned and the accuracy needed to break even; 3,034 stock-weeks with both sides measured |
| `fiveday/PREREGISTRATION.md`, `fiveday/README.md`, `fiveday/verdict.json` | Five-day direction test (pre-registered): can trend, reversal, the SPY rule or a gradient-boosted model call an S&P 500 stock's next-week direction well enough for V3? 2013-2026, 357,857 stock-weeks. No candidate passed |
| `trace/lineage.json` | Every agent trade traced back through promotion, stage 1, stage 2, sim 1, sim 2 and the entry morning |
| `databento_ledger.json` | Databento spend so far, against the $40 cap |
| `data_report.json` | Point-in-time data checks |

Book names: `lstm_copy` (what the sandbox agent would book: 15-min delayed quotes,
mid fills), `lstm_real` (real prices: ask in, bid out), `spy_rsi_real` (control,
no LSTM). Suffix `_0946` = entries wait until 9:46 ET; `_r1001` = 10:01 re-check
for tickers rejected only on the 20% spread rule. Proposed V3 changes (2026-10-22):
`_sp<N>` = entry spread cap N% instead of 20%; `_d2` = also sell at the bid at the
close of the 2nd trading day after entry.
