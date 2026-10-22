"""
Replay configuration. Everything that defines "what the live system does" is
read from the strategy modules themselves where possible; the numbers here are
the ones that only exist in the console settings or in this harness.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = ROOT / "strategy"
DATA_DIR = Path(os.environ.get("REPLAY_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("REPLAY_RESULTS_DIR", ROOT / "results"))

# The out-of-sample spread test (2026-10-22) uses its own, wider Sharadar
# subset; these env overrides leave the 2025 replay's defaults unchanged.
SHARADAR_DIR = Path(os.environ.get("REPLAY_SHARADAR_DIR", DATA_DIR / "sharadar"))
DBN_CACHE_DIR = DATA_DIR / "databento"

# ── Replay window ────────────────────────────────────────────────────────────
# Weekly decisions are made "over the weekend" using data through the last
# trading day of each week. The first decision (week ending 2025-01-03)
# governs entries from 2025-01-06; the last (week ending 2025-12-26) governs
# 2025-12-29 .. 2025-12-31. Open positions are managed into January 2026.
REPLAY_FIRST_ASOF = "2025-01-03"
REPLAY_LAST_ASOF = "2025-12-26"
AGENT_LAST_ENTRY_DAY = "2025-12-31"
AGENT_LAST_EXIT_DAY = "2026-01-30"

# Sharadar history needed: LSTM trains from 2018-01-01 (lstm_test.TRAIN_START,
# lstm_forecast start); the sim needs 365 + 60 days before each Friday.
DATA_START = os.environ.get("REPLAY_DATA_START", "2017-10-01")
DATA_END = os.environ.get("REPLAY_DATA_END", "2026-02-15")

# S&P 500 members at ANY point in this window form the price universe.
MEMBERSHIP_WINDOW = tuple(os.environ.get("REPLAY_MEMBERSHIP_WINDOW", "2024-12-01,2025-12-31").split(","))
# NYSE calendar range loaded by asof.nyse_schedule()
CALENDAR_START = os.environ.get("REPLAY_CAL_START", "2024-01-01")
CALENDAR_END = os.environ.get("REPLAY_CAL_END", "")          # "" = AGENT_LAST_EXIT_DAY

# ── Console settings at the time (app/pipeline/ui_settings.json, run state) ──
MAX_CANDIDATES = 25          # "Cap candidates" used on the 2026-08-07 run
MIN_TRADES = 1               # promotion: trades required in each harness
MAX_OPEN_SIM = 3             # sim max open (sidebar default = agent's value)

# ── Funnel change requested 2026-10-16 (differs from the live console) ──────
# Stage 2 -> 3 gate: a candidate's total intraday-backtest P&L must be at
# least this % of the fixed per-trade stake (intraday_backtest.BUDGET_PER_TRADE,
# $10,000), i.e. 200 -> >= $20,000 total across its trades.
STAGE2_MIN_GAIN_PCT = float(os.environ.get("STAGE2_MIN_GAIN_PCT", "200"))
# Stage 3 then runs the sim twice: on the gate survivors, then again on only
# the tickers that were positive in the first sim. Tickers positive in the
# second sim are promoted.
#
# Variants (2026-10-21): the minimum number of trades a ticker needs in EACH
# sim to count as positive. Stage 2 and sim 1 are shared; the variants split at
# the sim-1 filter. Name -> min trades.
SIM_MIN_TRADES_VARIANTS = {
    f"sim_min{n}": int(n)
    for n in os.environ.get("SIM_MIN_TRADES_VARIANTS", "1,3").split(",") if n.strip()
}
# Reuse the committed stage-2 results instead of re-running the intraday
# backtest (which needs the cached Databento bars).
REUSE_STAGE2 = os.environ.get("REUSE_STAGE2", "") == "1"

# ── Live agent (trading_agent_v3.py / paper_trading.py) ──────────────────────
STARTING_BALANCE = 5000.0    # agent comments describe a $5,000 paper balance
BASE_POSITION_PCT = 0.20
MAX_OPEN_POSITIONS = 3
MIN_TRADE_BUDGET = 10.0      # paper_trading.MIN_TRADE_BUDGET
SELECT_OPTION_MIN_BUDGET = 50.0
ENTRY_ALLOWED_WEEKDAYS = {0, 1, 2}
ENTRY_MIN_LSTM_MOVE_PCT = 1.0
ENTRY_MIN_DTE = 3
EARNINGS_SOON_DAYS = 7
EXIT_TRAIL_TRIGGER_PCT = 50.0
EXIT_TRAIL_GIVEBACK_PCT = 20.0
EXIT_HARD_CEILING_PCT = 200.0
EXIT_STOP_LOSS_PCT = -75.0
EXIT_FORCE_HOUR_ET = 13
COPY_FIXED_DELTA = 0.35      # paper_trading._calc_option_pnl default delta
MAX_SPREAD_PCT = 0.20
ASSUMED_MOVE_PCT = 0.02
MIN_DELTA = 0.35
RISK_FREE = 0.045

# Timing model for the market-open scan. The live loop runs the exit engine,
# then trains one LSTM per universe ticker before deploying anything; on the
# PC that is ~75-80 s per ticker. Tickers skipped on the earnings check never
# reach the LSTM.
SCAN_SECONDS_PER_TICKER = 80
SCAN_SECONDS_EARNINGS_SKIP = 2

# Tradier sandbox quotes are 15-minute delayed. Copy mode reproduces that.
COPY_QUOTE_DELAY_MIN = 15

# ── Databento ────────────────────────────────────────────────────────────────
DATABENTO_BUDGET_USD = float(os.environ.get("DATABENTO_BUDGET_USD", "70"))
EQUITY_DATASETS = ["EQUS.MINI", "XNAS.ITCH"]   # first one covering the window wins
OPTIONS_DATASET = "OPRA.PILLAR"

# ── Reproducibility ──────────────────────────────────────────────────────────
# The live system never seeds TensorFlow, so two live runs of the same day can
# disagree. The replay seeds every fit from (ticker, date) so reruns match.
SEED_BASE = 20250103
