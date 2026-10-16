# Prometheus replay: the weekly funnel, walked forward through 2025

This repo replays a weekly options-trading workflow over every week of 2025,
using only data that existed at each point in time, to answer one question:
**if the system had run exactly as designed for a year, what would have
happened?**

The live system has two halves:

1. **A weekend research funnel** that picks which tickers the agent may trade:
   an LSTM accuracy sweep over the S&P 500 → an intraday options backtest → a
   portfolio simulation → promotion of tickers positive in both.
2. **A live agent** that trades the promoted whitelist Monday to Wednesday:
   per-ticker LSTM direction filtered by SPY's trend, ranked by
   signal-to-noise, buying 2-3 week calls or puts and exiting on a fixed rule
   stack (stop −75%, ceiling +200%, trailing stop, Thursday-before-expiry exit).

The replay runs both halves for 52 weekends and one continuous trading book.

**Funnel change from the live console (2026-10-16).** Instead of "positive in
both the backtest and the sim", the replay now uses: stage-2 total P&L ≥ 200%
of the $10,000 per-trade stake (≥ $20,000) → sim on those survivors → sim again
on only the tickers positive in the first sim → tickers positive in the second
sim are promoted. The threshold is the `stage2_min_gain_pct` input of workflow 3.

## What is reused and what is ported

`strategy/` holds the live repo's research modules **unmodified**:
`lstm_test.py`, `lstm_forecast.py`, `portfolio_sim_v3.py`,
`intraday_backtest.py`. The replay swaps their data sources at import time
(`replay/shims.py`) instead of editing them, the same way the live console
injects its whitelist.

The live agent's entry and exit code calls a broker sandbox and Google Sheets
directly, so its rules are ported line for line into `replay/agent.py` with
references to the original functions.

## Point-in-time discipline

| Input | Source in the replay | What the live system used |
|---|---|---|
| S&P 500 universe for the sweep | Sharadar `sp500`, membership **as of that Friday** (quarterly snapshot + adds/removes) | Today's Wikipedia list |
| Daily prices | Sharadar `stocks` / `funds`, cut at the as-of date and re-anchored so adjustments match what the source showed then | yfinance / broker history |
| Earnings dates | Sharadar `events`, 8-K Item 2.02 filings | yfinance calendar |
| Intraday bars (stage 2 and agent) | Databento 1-minute bars | yfinance 15-minute bars / broker quotes |
| Option chains and quotes (agent) | Databento OPRA definitions, 1-minute NBBO, trades | broker sandbox chain |

Every LSTM fit is seeded from (ticker, date) so the replay is reproducible.
The live system is not seeded, so two live runs of the same day can differ.

## Agent modes

| Mode | Exit triggers | Fills | Quotes |
|---|---|---|---|
| `copy` | the agent's own P&L estimate (fixed 0.35 delta on the stock move, no time decay) | buy at mid; sell at the option's last trade, else bid | 15-minute delayed, like the sandbox |
| `real` | the held option's real mid | buy at ask, sell at bid | real time |

`copy` answers *what would my agent have done and booked*. `real` answers
*what would the market actually have paid for those decisions*.

Selector `spy_rsi` is the control: same whitelist and execution, but every
ticker trades in SPY's trend direction ranked by RSI (the sim's rule), with no
LSTM. If `lstm` does not beat `spy_rsi`, the LSTM is not adding anything.

## Running it (GitHub Actions, in order)

Secrets: `SHARADAR_API_KEY`, `DATABENTO_API_KEY`. Every workflow is manual
(`workflow_dispatch`); nothing runs on push or on pull requests.

1. **Prepare Sharadar data**: downloads the bulk tables once, keeps a small
   subset in the Actions cache. Raw data is never committed.
2. **Stage 1 LSTM sweep**: run once per quarter (1, 2, 3, 4). Each run is 13
   weeks × 6 shards of ~84 tickers.
3. **Stages 2-4**: run first with `quote_only` ticked to see the Databento
   price of the stage-2 bars, then again unticked to buy them and run the
   backtest, sim and promotion for every week.
4. **Agent forecasts**: the morning LSTM fits for every whitelisted ticker on
   every Mon/Tue/Wed.
5. **Agent replay**: the continuous book in each mode. Databento pulls are
   priced first and stop hard at the budget.

Results land in `results/`: per-week sweep/backtest/sim/promotion files,
`whitelist_timeline.csv`, and `results/agent/<selector>_<mode>/` with
`trades.csv`, `daily_equity.csv`, `entry_log.csv`, `summary.json`.

Actions caches are evicted after 7 days without use, so run the steps within a
week of each other (or rerun step 1).

## Known approximations

- The agent's scan takes ~80 s per ticker (one LSTM fit each); entries happen
  at the open plus that delay, and exits are not checked during the scan,
  matching the single-threaded live loop.
- OPRA has no greeks; delta comes from Black-Scholes at the mid's implied vol.
  The live chain's greeks come from the broker's vendor.
- Earnings dates are 8-K filing dates, which almost always equal the release
  date.
- The sweep's "Cap candidates" is 25 and promotion needs ≥1 trade in each
  harness, matching the console run of 2026-08-07. When nothing is promoted
  the previous whitelist carries over; the first weeks start empty.
- The live agent's daily history may or may not include the current session;
  the replay assumes it ends at the prior close.

## Smoke test

`python tests/smoke.py` runs the whole chain on synthetic data with fake
Databento responses and one-epoch models. No keys or network needed.
