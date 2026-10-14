"""
╔══════════════════════════════════════════════════════════════════════════╗
║      PROMETHEUS v3 — PORTFOLIO SIMULATOR                                ║
║                                                                          ║
║  Replays v3 entry/exit logic over a 365-day historical window using    ║
║  Black-Scholes-modeled option premiums and yfinance historical prices. ║
║                                                                          ║
║  Honest limits:                                                          ║
║    - BS pricing does not model IV crush, bid/ask, or smile.            ║
║    - Earnings veto uses yfinance.earnings_dates (fail-open on None).   ║
║    - Results are useful for RELATIVE comparisons, not absolute predict. ║
║                                                                          ║
║  Output: sim_output/month_XX_YYYY-MM.json per month + annual_summary.   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import math
import gc
import random
import pickle
import logging
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PortfolioSimV3")

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — mirrors trading_agent_v3.py exactly
# ═════════════════════════════════════════════════════════════════════════════

# Top-17 whitelist from the LSTM qualification screen (2026-08-03), ranked by SCORE.
# Revert to the v3 universe (GOOG/MSFT/AMZN/... ) to return to baseline.
TICKER_UNIVERSE = [
      "UNH", 
]
VOLATILE_TICKERS = set()   # no volatile-class tickers in the whitelist

# Sim window
SIM_DAYS              = 365         # calendar days back from today
MONTH_BATCH_DAYS      = 30          # output + RAM flush every N sim days
STARTING_BALANCE      = 5000.0

# Position sizing (v3 current)
BASE_POSITION_PCT     = 0.20
MAX_OPEN_POSITIONS    = 2
MIN_TRADE_BUDGET      = 50.0

# Entry filters (v3 current)
ENTRY_ALLOWED_WEEKDAYS = {0, 1, 2}   # Mon/Tue/Wed — restored for out-of-sample test
ENTRY_RSI_PERIOD       = 14
EARNINGS_VETO_DAYS     = 7

# Option selection (v3 current)
# Expiry is snapped to the (DTE_PREFER_HI // 7)-th Friday strictly after the
# sim date, so actual DTE lands in the 15-21 day range (2-3 week Friday).
DTE_PREFER_LO         = 14
DTE_PREFER_HI         = 21
DTE_ACCEPT_LO         = 5
DTE_ACCEPT_HI         = 21
STRIKE_OTM_PCT_CALL_LO = 1.00
STRIKE_OTM_PCT_CALL_HI = 1.03
STRIKE_OTM_PCT_PUT_LO  = 0.97
STRIKE_OTM_PCT_PUT_HI  = 1.001
STRIKE_FALLBACK_CALL_HI = 1.05
STRIKE_FALLBACK_PUT_LO  = 0.95
MIN_DELTA             = 0.35
ASSUMED_MOVE_PCT      = 0.02

# Exit rules (v3 current)
EXIT_TRAIL_TRIGGER_PCT   = 50.0
EXIT_TRAIL_GIVEBACK_PCT  = 20.0
EXIT_HARD_CEILING_PCT    = 200.0  # was 100.0 — sim experiment 2026-05-28
EXIT_STOP_LOSS_PCT       = -75.0
EXIT_BLEED_PCT           = -30.0   # unused — Rule 4 (bleed stop) removed 2026-07-13
EXIT_NO_MOVE_PCT         = 1.0     # unused — Rule 4 (bleed stop) removed 2026-07-13
EXIT_SAFETY_NET_LIFE_PCT = 0.40    # unused — Rule 5 (safety net) removed 2026-07-13
EXIT_FORCE_WEEKDAY       = 3        # Thursday
EXIT_FORCE_HOUR_ET       = 13

# Out-of-sample window shift. 0 = recent year [today-365, today].
# 365 = prior year [today-730, today-365]. Set by sweep scripts at runtime.
OOS_SHIFT_DAYS        = 0

# Pricing model
RISK_FREE_RATE        = 0.045
IV_LOOKBACK_DAYS      = 30          # realized vol window for IV proxy

# Output
OUTPUT_DIR            = "sim_output"
CACHE_DIR             = "data_cache"   # pickled load_window data, keyed on anchor


@dataclass
class SimConfig:
    """
    Injected config for programmatic sim runs (sweep harnesses). Every
    field defaults to the CURRENT module-level constant — defaults are
    read at instantiation time (default_factory), so sweep scripts that
    set module globals at runtime (e.g. OOS_SHIFT_DAYS) still work.
    """
    ticker_universe:          list  = field(default_factory=lambda: list(TICKER_UNIVERSE))
    volatile_tickers:         set   = field(default_factory=lambda: set(VOLATILE_TICKERS))
    sim_days:                 int   = field(default_factory=lambda: SIM_DAYS)
    month_batch_days:         int   = field(default_factory=lambda: MONTH_BATCH_DAYS)
    starting_balance:         float = field(default_factory=lambda: STARTING_BALANCE)
    base_position_pct:        float = field(default_factory=lambda: BASE_POSITION_PCT)
    max_open_positions:       int   = field(default_factory=lambda: MAX_OPEN_POSITIONS)
    min_trade_budget:         float = field(default_factory=lambda: MIN_TRADE_BUDGET)
    entry_allowed_weekdays:   set   = field(default_factory=lambda: set(ENTRY_ALLOWED_WEEKDAYS))
    entry_rsi_period:         int   = field(default_factory=lambda: ENTRY_RSI_PERIOD)
    earnings_veto_days:       int   = field(default_factory=lambda: EARNINGS_VETO_DAYS)
    dte_prefer_lo:            int   = field(default_factory=lambda: DTE_PREFER_LO)
    dte_prefer_hi:            int   = field(default_factory=lambda: DTE_PREFER_HI)
    dte_accept_lo:            int   = field(default_factory=lambda: DTE_ACCEPT_LO)
    dte_accept_hi:            int   = field(default_factory=lambda: DTE_ACCEPT_HI)
    strike_otm_pct_call_lo:   float = field(default_factory=lambda: STRIKE_OTM_PCT_CALL_LO)
    strike_otm_pct_call_hi:   float = field(default_factory=lambda: STRIKE_OTM_PCT_CALL_HI)
    strike_otm_pct_put_lo:    float = field(default_factory=lambda: STRIKE_OTM_PCT_PUT_LO)
    strike_otm_pct_put_hi:    float = field(default_factory=lambda: STRIKE_OTM_PCT_PUT_HI)
    strike_fallback_call_hi:  float = field(default_factory=lambda: STRIKE_FALLBACK_CALL_HI)
    strike_fallback_put_lo:   float = field(default_factory=lambda: STRIKE_FALLBACK_PUT_LO)
    min_delta:                float = field(default_factory=lambda: MIN_DELTA)
    assumed_move_pct:         float = field(default_factory=lambda: ASSUMED_MOVE_PCT)
    exit_trail_trigger_pct:   float = field(default_factory=lambda: EXIT_TRAIL_TRIGGER_PCT)
    exit_trail_giveback_pct:  float = field(default_factory=lambda: EXIT_TRAIL_GIVEBACK_PCT)
    exit_hard_ceiling_pct:    float = field(default_factory=lambda: EXIT_HARD_CEILING_PCT)
    exit_stop_loss_pct:       float = field(default_factory=lambda: EXIT_STOP_LOSS_PCT)
    exit_bleed_pct:           float = field(default_factory=lambda: EXIT_BLEED_PCT)
    exit_no_move_pct:         float = field(default_factory=lambda: EXIT_NO_MOVE_PCT)
    exit_safety_net_life_pct: float = field(default_factory=lambda: EXIT_SAFETY_NET_LIFE_PCT)
    exit_force_weekday:       int   = field(default_factory=lambda: EXIT_FORCE_WEEKDAY)
    exit_force_hour_et:       int   = field(default_factory=lambda: EXIT_FORCE_HOUR_ET)
    oos_shift_days:           int   = field(default_factory=lambda: OOS_SHIFT_DAYS)
    risk_free_rate:           float = field(default_factory=lambda: RISK_FREE_RATE)
    iv_lookback_days:         int   = field(default_factory=lambda: IV_LOOKBACK_DAYS)
    # Sweep knobs (no module-level constant; defaults preserve current behavior)
    max_new_positions_per_day: int  = 99      # per-day deploy cap; 99 = fill all open slots
    spy_ma_period:            int   = 20      # SPY trend MA period
    rank_mode:                str   = "rsi"   # "rsi" | "inverse_rsi" | "random"
    spy_mode:                 str   = "trend" # "trend" | "always_bullish" | "random"
    random_seed:              int   = 42
    # Reproducibility: pin the END of the sim window to a fixed date so
    # results don't drift with the calendar. None = today (current behavior).
    # Window = [anchor - oos_shift_days - sim_days, anchor - oos_shift_days]
    anchor_date:              str | None = None   # "YYYY-MM-DD"
    use_cache:                bool  = True        # load_window disk cache
    # Execution-cost modeling (opt-in). RELATIVE comparator only — spread
    # model is a point-in-time snapshot, not historical friction.
    apply_costs:              bool  = False
    spread_model_path:        str | None = "spread_model.json"
    commission_per_contract:  float = 0.35   # per contract, charged on entry AND exit


def _anchor(cfg: SimConfig) -> date:
    return date.fromisoformat(cfg.anchor_date) if cfg.anchor_date else date.today()


# ═════════════════════════════════════════════════════════════════════════════
#  PRICING HELPERS (Black-Scholes — replaceable later with Polygon)
# ═════════════════════════════════════════════════════════════════════════════

def black_scholes_price(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str = "call") -> float:
    """Black-Scholes premium. T is in years. Returns 0 on bad inputs."""
    try:
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if option_type == "call":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(0.0, float(price))
    except Exception:
        return 0.0


def black_scholes_delta(S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str = "call") -> float:
    """Black-Scholes delta. Returns absolute value (0 to 1)."""
    try:
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
            return 0.0
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        if option_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(abs(norm.cdf(d1) - 1.0))
    except Exception:
        return 0.0


def estimate_iv(prices: pd.Series, cfg: SimConfig | None = None) -> float:
    """Annualized realized vol over the tail of `prices` as an IV proxy."""
    if cfg is None:
        cfg = SimConfig()
    try:
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return 0.20
        rv = float(returns.tail(cfg.iv_lookback_days).std() * math.sqrt(252))
        return max(0.10, rv)
    except Exception:
        return 0.20


# ── Execution-cost model (opt-in via cfg.apply_costs) ────────────────────────

_SPREAD_MODEL_CACHE: dict = {}   # path -> parsed spread_model.json


def _load_spread_model(cfg: SimConfig) -> dict:
    """
    Lazily load spread_model.json (once per path, module-level cache).
    Raises if apply_costs is on and the file is missing — never silently
    falls back to a frictionless run.
    """
    path = cfg.spread_model_path
    if not path:
        raise FileNotFoundError(
            "apply_costs=True but spread_model_path is None — "
            "set it or run calibrate_option_spreads.py"
        )
    cached = _SPREAD_MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"apply_costs=True but spread model '{path}' not found — "
            f"run calibrate_option_spreads.py first (refusing to run frictionless)"
        )
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    _SPREAD_MODEL_CACHE[path] = model
    return model


def spread_pct_for(moneyness: float, dte: float, model: dict) -> float:
    """
    Median spread_pct for the (moneyness bin x dte bin) bucket from
    spread_model.json; falls back to the overall median for anything
    outside the bins or in an empty bucket.
    """
    fallback = float((model.get("fallback") or {}).get("median_spread_pct") or 0.0)
    meta = model.get("metadata") or {}
    for m_lo, m_hi in meta.get("moneyness_bins", []):
        if not (m_lo <= moneyness < m_hi):
            continue
        for d_lo, d_hi in meta.get("dte_bins", []):
            if not (d_lo <= dte <= d_hi):
                continue
            key = f"moneyness_{m_lo:.2f}-{m_hi:.2f}|dte_{d_lo}-{d_hi}"
            med = ((model.get("buckets") or {}).get(key) or {}).get("median_spread_pct")
            if med is not None:
                return float(med)
    return fallback


# ═════════════════════════════════════════════════════════════════════════════
#  DATA FETCH + INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

def fetch_historical_data(ticker: str, days_back: int,
                          oos_shift_days: int | None = None,
                          cfg: SimConfig | None = None) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV via yfinance, return DataFrame indexed by date."""
    try:
        if cfg is None:
            cfg = SimConfig()
        # Out-of-sample shift: explicit arg wins; falls back to the
        # module-level OOS_SHIFT_DAYS global (sweep scripts set it at runtime).
        shift = OOS_SHIFT_DAYS if oos_shift_days is None else oos_shift_days
        end = _anchor(cfg) - timedelta(days=shift)
        # Add 60-day buffer so RSI/IV have warmup data before sim start
        start = end - timedelta(days=days_back + 60)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            log.warning(f"[{ticker}] yfinance returned empty data")
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        log.warning(f"[{ticker}] fetch failed: {e}")
        return None


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI on a close-price series. Returns full series."""
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    loss_safe = loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + gain / loss_safe))
    return rsi


def fetch_earnings_dates(ticker: str) -> list:
    """
    Fetch historical + upcoming earnings dates for a ticker.
    Returns a list of date objects. Empty list on failure (fail-open).
    """
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or (hasattr(ed, "empty") and ed.empty):
            log.warning(f"[{ticker}] earnings_dates empty — earnings veto disabled for this ticker")
            return []
        dates = []
        for idx in ed.index:
            try:
                dates.append(pd.Timestamp(idx).date())
            except Exception:
                continue
        return dates
    except Exception as e:
        log.warning(f"[{ticker}] earnings_dates fetch failed ({e}) — veto disabled for this ticker")
        return []


def has_earnings_within(ticker_earnings: list, as_of: date, days: int = 7) -> bool:
    """Check if any earnings date falls within [as_of, as_of + days]."""
    if not ticker_earnings:
        return False  # fail-open: no data = no veto
    cutoff = as_of + timedelta(days=days)
    return any(as_of <= ed <= cutoff for ed in ticker_earnings)


# ═════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OpenTrade:
    ticker:         str
    direction:      str          # "Bullish" or "Bearish"
    opt_type:       str          # "call" or "put"
    strike:         float
    entry_premium:  float        # per-share
    contracts:      int
    total_cost:     float        # contracts * entry_premium * 100
    entry_date:     date
    entry_price:    float        # underlying price at entry
    expiry_date:    date
    iv:             float        # IV used for pricing this contract
    entry_delta:    float
    is_volatile:    bool
    peak_pnl_pct:   float = 0.0  # for trail-stop tracking


@dataclass
class PortfolioState:
    balance:        float = STARTING_BALANCE
    open_trades:    list  = field(default_factory=list)   # list[OpenTrade]
    balance_curve:  list  = field(default_factory=list)   # (date, balance) — cash only
    equity_curve:   list  = field(default_factory=list)   # (date, balance + open MV)
    # Per-month buffers — flushed to disk + cleared between months
    month_closed_trades: list = field(default_factory=list)
    month_entries:       list = field(default_factory=list)
    month_skips:         list = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY (stub — filled in by Prompt D)
# ═════════════════════════════════════════════════════════════════════════════

def run_sim(cfg: SimConfig, data: dict | None = None,
            write_output: bool = True) -> dict:
    """
    Programmatic sim entry for sweep harnesses.

    - `data` is {"spy_df":..., "ticker_data":..., "earnings_data":...}
      (from load_window); None loads it fresh via load_all_data.
    - write_output=False skips all disk writes (month JSONs + annual
      summary) and accumulates closed trades in memory only.

    Returns a metrics dict: n_trades, win_rate, expectancy, total_pnl,
    return_pct, max_dd_pct, ending_balance, exit_reason_counts,
    per_ticker_pnl, closed_trades (+ open_at_end). Empty dict if no
    ticker data could be loaded.
    """
    if write_output:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Wipe stale output from any previous run (month_*.json + annual_summary.json).
        # Orphaned files with different month labels were silently leaking into the
        # diagnostic and inflating trade counts. Clear the directory every run.
        for _f in os.listdir(OUTPUT_DIR):
            if _f.endswith(".json"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, _f))
                except OSError as _e:
                    log.warning(f"could not remove stale output {_f}: {_e}")

    # ── Load all data upfront (or reuse the injected dict) ───────────────
    if data is None:
        spy_df, ticker_data, earnings_data = load_all_data(cfg.sim_days, cfg)
    else:
        spy_df        = data["spy_df"]
        ticker_data   = data["ticker_data"]
        earnings_data = data["earnings_data"]
    if not ticker_data:
        log.error("No ticker data loaded — aborting.")
        return {}

    # ── Build sim day list (weekdays only, within sim_days window) ───────
    today    = _anchor(cfg) - timedelta(days=cfg.oos_shift_days)   # OOS shift — match fetch window
    sim_start = today - timedelta(days=cfg.sim_days)
    all_days = []
    d = sim_start
    while d <= today:
        if d.weekday() < 5:   # Mon-Fri only
            all_days.append(d)
        d += timedelta(days=1)
    log.info(f"Sim day count: {len(all_days)} trading days "
             f"({sim_start} -> {today})")

    # ── State ────────────────────────────────────────────────────────────
    state = PortfolioState(balance=cfg.starting_balance)

    # One RNG per run so rank_mode/spy_mode randomness is independently
    # reproducible across runs (never the global random module).
    rng = random.Random(cfg.random_seed)

    # ── Main loop ────────────────────────────────────────────────────────
    month_idx          = 1
    days_in_this_batch = 0
    batch_start_date   = all_days[0] if all_days else today
    all_closed_trades: list = []   # in-memory copy of every closed trade
    exit_engine_errors    = 0      # absorbed exceptions — sweep corruption guard
    entry_pipeline_errors = 0
    unrevaluable_marks    = 0      # daily equity marks that fell back to 0.0

    def _collect_and_flush(midx: int, mlabel: str) -> None:
        # Capture closed trades before flush_month clears the buffer.
        all_closed_trades.extend(state.month_closed_trades)
        if write_output:
            flush_month(state, midx, mlabel, OUTPUT_DIR)
        else:
            state.month_entries       = []
            state.month_skips         = []
            state.month_closed_trades = []

    for sim_date in all_days:
        # Record balance curve once per sim day
        state.balance_curve.append((sim_date, round(state.balance, 2)))

        # Record mark-to-market equity at the SAME point so the two curves
        # are directly comparable. Reuses revalue_trade (the exit engine's
        # pricing path); an unrevaluable position marks at 0.0 and is
        # counted — this loop must never raise.
        equity_for_day = state.balance
        for trade in state.open_trades:
            try:
                df = ticker_data.get(trade.ticker)
                valuation = revalue_trade(trade, sim_date, df, cfg) if df is not None else None
                if valuation is not None and math.isfinite(valuation["current_premium"]):
                    equity_for_day += valuation["current_premium"] * 100 * trade.contracts
                else:
                    unrevaluable_marks += 1
            except Exception:
                unrevaluable_marks += 1
        state.equity_curve.append((sim_date, round(equity_for_day, 2)))

        # Exit engine first (mirrors live agent: exits run before entries
        # so a closed position frees the slot for a new entry)
        try:
            run_exit_engine_sim(state, sim_date, ticker_data, cfg)
        except Exception:
            exit_engine_errors += 1
            log.exception(f"[{sim_date}] exit engine raised — continuing")

        # Entry pipeline
        try:
            try_market_open_entry_sim(
                state, sim_date, spy_df, ticker_data, earnings_data, cfg, rng,
            )
        except Exception:
            entry_pipeline_errors += 1
            log.exception(f"[{sim_date}] entry pipeline raised — continuing")

        days_in_this_batch += 1

        # ── Monthly flush ────────────────────────────────────────────────
        if days_in_this_batch >= cfg.month_batch_days:
            month_label = f"{batch_start_date.year}-{batch_start_date.month:02d}"
            _collect_and_flush(month_idx, month_label)
            month_idx          += 1
            days_in_this_batch  = 0
            batch_start_date    = sim_date + timedelta(days=1)

    # ── Final partial month ──────────────────────────────────────────────
    if days_in_this_batch > 0 or state.month_entries or state.month_closed_trades:
        month_label = f"{batch_start_date.year}-{batch_start_date.month:02d}"
        _collect_and_flush(month_idx, month_label)

    # ── Annual summary (disk) ────────────────────────────────────────────
    if write_output:
        build_annual_summary(
            OUTPUT_DIR,
            state.balance_curve,
            cfg.starting_balance,
            state.balance,
        )

    # ── Mark remaining open positions to market (final sim day) ──────────
    # Metrics-only: state.balance and all pre-existing fields are untouched.
    # Reuses revalue_trade (the exit engine's pricing path); a position that
    # cannot be revalued is marked at 0.0 and counted in unrevaluable_open.
    final_day = all_days[-1] if all_days else today
    open_market_value = 0.0
    open_cost_basis   = 0.0
    unrevaluable_open = 0
    for trade in state.open_trades:
        open_cost_basis += trade.total_cost
        mv = 0.0
        try:
            df = ticker_data.get(trade.ticker)
            valuation = revalue_trade(trade, final_day, df, cfg) if df is not None else None
            if valuation is not None and math.isfinite(valuation["current_premium"]):
                mv = valuation["current_premium"] * 100 * trade.contracts
            else:
                unrevaluable_open += 1
        except Exception:
            unrevaluable_open += 1
            log.exception(f"[{trade.ticker}] end-of-sim mark-to-market failed — valued at 0")
        open_market_value += mv

    unrealized_pnl = open_market_value - open_cost_basis
    equity_at_end  = state.balance + open_market_value

    # ── Metrics (same formulas + rounding as build_annual_summary) ───────
    n_trades = len(all_closed_trades)
    wins   = [t for t in all_closed_trades if t["pnl_dollar"] > 0]
    losses = [t for t in all_closed_trades if t["pnl_dollar"] < 0]

    win_rate  = (len(wins) / n_trades * 100) if n_trades else 0.0
    avg_win   = (sum(t["pnl_dollar"] for t in wins)   / len(wins))   if wins   else 0.0
    avg_loss  = (sum(t["pnl_dollar"] for t in losses) / len(losses)) if losses else 0.0
    total_pnl = sum(t["pnl_dollar"] for t in all_closed_trades)
    if n_trades:
        wr = len(wins) / n_trades
        expectancy = (wr * avg_win) + ((1 - wr) * avg_loss)
    else:
        expectancy = 0.0

    max_dd_pct = 0.0
    peak = cfg.starting_balance
    for _, bal in state.balance_curve:
        if bal > peak:
            peak = bal
        dd_pct = (bal - peak) / peak * 100 if peak > 0 else 0.0
        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct

    # Same peak/trough formula, on the mark-to-market equity curve
    max_dd_equity_pct = 0.0
    peak_eq = cfg.starting_balance
    for _, eq in state.equity_curve:
        if eq > peak_eq:
            peak_eq = eq
        dd_pct = (eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0
        if dd_pct < max_dd_equity_pct:
            max_dd_equity_pct = dd_pct

    exit_reason_counts = {}
    for t in all_closed_trades:
        key = t["exit_reason"].split(" ")[0] if t.get("exit_reason") else "Unknown"
        exit_reason_counts[key] = exit_reason_counts.get(key, 0) + 1

    per_ticker = {}
    for t in all_closed_trades:
        tk = t["ticker"]
        if tk not in per_ticker:
            per_ticker[tk] = {"trades": 0, "wins": 0, "pnl": 0.0}
        per_ticker[tk]["trades"] += 1
        if t["pnl_dollar"] > 0:
            per_ticker[tk]["wins"] += 1
        per_ticker[tk]["pnl"] += t["pnl_dollar"]
    for tk, stats in per_ticker.items():
        stats["pnl"]      = round(stats["pnl"], 2)
        stats["win_rate"] = round(stats["wins"] / stats["trades"] * 100, 1) if stats["trades"] else 0.0

    return {
        "n_trades":           n_trades,
        "win_rate":           round(win_rate, 2),
        "expectancy":         round(expectancy, 2),
        "total_pnl":          round(total_pnl, 2),
        "return_pct":         round((state.balance - cfg.starting_balance) / cfg.starting_balance * 100, 2),
        "max_dd_pct":         round(max_dd_pct, 2),
        "ending_balance":     round(state.balance, 2),
        "exit_reason_counts": exit_reason_counts,
        "per_ticker_pnl":     per_ticker,
        "closed_trades":      all_closed_trades,
        "open_at_end":        len(state.open_trades),
        "exit_engine_errors":    exit_engine_errors,
        "entry_pipeline_errors": entry_pipeline_errors,
        # Mark-to-market equity view (open positions valued on the final day)
        "open_market_value":   round(open_market_value, 2),
        "open_cost_basis":     round(open_cost_basis, 2),
        "unrealized_pnl":      round(unrealized_pnl, 2),
        "equity_at_end":       round(equity_at_end, 2),
        "return_pct_equity":   round((equity_at_end - cfg.starting_balance) / cfg.starting_balance * 100, 2),
        "total_pnl_incl_open": round(total_pnl + unrealized_pnl, 2),
        "unrevaluable_open":   unrevaluable_open,
        # Daily mark-to-market equity curve (same shape as balance_curve)
        "max_dd_equity_pct":   round(max_dd_equity_pct, 2),
        "unrevaluable_marks":  unrevaluable_marks,
        "equity_curve":        state.equity_curve,
    }


def run_portfolio_sim_v3() -> None:
    """
    Top-level sim entry. Replays v3 entry/exit over SIM_DAYS calendar
    days, batched by month with RAM flush between months. Thin wrapper
    over run_sim(SimConfig(), data=None, write_output=True).
    """
    cfg = SimConfig()

    log.info("=" * 70)
    log.info("  PROMETHEUS v3 — PORTFOLIO SIM START")
    log.info(f"  Window: {cfg.sim_days} calendar days  |  Start: ${cfg.starting_balance:.2f}")
    log.info(f"  Tickers: {len(cfg.ticker_universe)}  |  Size: {cfg.base_position_pct*100:.0f}%  "
             f"|  Max open: {cfg.max_open_positions}")
    log.info("=" * 70)

    metrics = run_sim(cfg, data=None, write_output=True)
    if not metrics:
        return

    # ── Console summary ──────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("  SIM COMPLETE")
    log.info(f"  Trades: {metrics['n_trades']}  |  "
             f"Win rate: {metrics['win_rate']:.1f}%  |  "
             f"Expectancy: ${metrics['expectancy']:.2f}")
    log.info(f"  Total P&L: ${metrics['total_pnl']:.2f}  |  "
             f"Return: {metrics['return_pct']:.1f}%  |  "
             f"Max DD: {metrics['max_dd_pct']:.1f}%")
    log.info(f"  Max DD (cash): {metrics['max_dd_pct']:.1f}%  |  "
             f"Max DD (equity): {metrics['max_dd_equity_pct']:.1f}%")
    log.info(f"  Ending balance: ${metrics['ending_balance']:.2f}  |  "
             f"Open at end: {metrics['open_at_end']}")
    log.info(f"  Equity: ${metrics['equity_at_end']:.2f} "
             f"(open MV ${metrics['open_market_value']:.2f} on "
             f"{metrics['open_at_end']} positions)  |  "
             f"Return(equity): {metrics['return_pct_equity']:.1f}%")
    log.info("  Exit reasons:")
    for reason, count in sorted(metrics["exit_reason_counts"].items(),
                                 key=lambda x: -x[1]):
        log.info(f"    {reason}: {count}")
    log.info("=" * 70)

    # ── Auto-run the diagnostic on the output we just wrote ──────────────
    print()
    print("=" * 70)
    print("  RUNNING DIAGNOSTIC ON SIM OUTPUT")
    print("=" * 70)
    try:
        import sim_diagnostic
        sim_diagnostic.main()
    except Exception as e:
        log.warning(f"Diagnostic failed to run: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY PIPELINE — SPY trend + per-ticker evaluation
# ═════════════════════════════════════════════════════════════════════════════

def get_spy_trend_sim(spy_df: pd.DataFrame, sim_date: date,
                      cfg: SimConfig | None = None) -> str:
    """
    SPY spy_ma_period-day MA trend as of sim_date. Mirrors
    trading_agent_v3.get_market_trend.
    Uses closes up to and INCLUDING sim_date (no lookahead).
    Returns 'Bullish', 'Bearish', or 'Neutral'.
    """
    if cfg is None:
        cfg = SimConfig()
    try:
        # Slice the DataFrame to rows up to sim_date
        mask = spy_df.index.date <= sim_date
        window = spy_df[mask]
        if len(window) < cfg.spy_ma_period + 1:   # warmup guard: need period+1 rows
            return "Neutral"
        close = window["Close"]
        ma = float(close.rolling(cfg.spy_ma_period).mean().iloc[-1])
        current = float(close.iloc[-1])
        return "Bullish" if current > ma else "Bearish"
    except Exception as e:
        log.warning(f"SPY trend failed at {sim_date}: {e}")
        return "Neutral"


def evaluate_ticker_sim(
    ticker: str,
    direction: str,
    sim_date: date,
    ticker_df: pd.DataFrame,
    ticker_earnings: list,
    cfg: SimConfig | None = None,
) -> dict:
    """
    Per-ticker evaluation for a given sim date. Mirrors
    trading_agent_v3.evaluate_ticker but reads from pre-loaded data.

    Returns a pass or skip dict, same shape as the live agent.
    """
    if cfg is None:
        cfg = SimConfig()
    # Live price = close on sim_date (or most recent prior trading day if
    # sim_date is a non-trading day — searchsorted handles this).
    try:
        mask = ticker_df.index.date <= sim_date
        window = ticker_df[mask]
        if window.empty:
            return {"passed": False, "ticker": ticker,
                    "skip_reason": "no price data up to sim_date"}
        current_price = float(window["Close"].iloc[-1])
    except Exception as e:
        return {"passed": False, "ticker": ticker,
                "skip_reason": f"price lookup failed: {e}"}

    # Earnings veto (fail-open if no data). Horizon = the actual prospective
    # hold: entry -> Thursday before this sim_date's expiry Friday (the real
    # last day of exposure), floored at cfg.earnings_veto_days so the veto
    # can never be looser than the old fixed-7-day behavior.
    expiry_dt, _ = _select_expiry_friday(sim_date, cfg)
    veto_days = max((expiry_dt - timedelta(days=1) - sim_date).days,
                    cfg.earnings_veto_days)
    if has_earnings_within(ticker_earnings, sim_date, veto_days):
        return {"passed": False, "ticker": ticker,
                "skip_reason": "earnings within hold window — IV crush risk"}

    # RSI
    try:
        rsi_series = compute_rsi(window["Close"], period=cfg.entry_rsi_period)
        rsi_val = float(rsi_series.iloc[-1])
        if rsi_val != rsi_val:  # NaN check
            return {"passed": False, "ticker": ticker,
                    "skip_reason": "RSI is NaN"}
    except Exception as e:
        return {"passed": False, "ticker": ticker,
                "skip_reason": f"RSI computation failed: {e}"}

    # RSI score (same as live agent)
    if direction == "Bullish":
        rsi_score = 100.0 - rsi_val
    else:
        rsi_score = rsi_val

    return {
        "passed":        True,
        "ticker":        ticker,
        "direction":     direction,
        "rsi":           rsi_val,
        "rsi_score":     rsi_score,
        "current_price": current_price,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY PIPELINE — option selection
# ═════════════════════════════════════════════════════════════════════════════

def _generate_strikes(current: float, lo_mult: float, hi_mult: float) -> list:
    """Generate synthetic $0.50-spaced strikes between current*lo and current*hi."""
    lo = current * lo_mult
    hi = current * hi_mult
    # Round to nearest $0.50 grid
    start = math.floor(lo * 2) / 2
    end   = math.ceil(hi * 2) / 2
    strikes = []
    k = start
    while k <= end + 1e-9:
        if lo <= k <= hi:
            strikes.append(round(k, 2))
        k += 0.5
    return strikes


def _select_expiry_friday(sim_date: date, cfg: SimConfig | None = None) -> tuple:
    """
    Return (expiry_friday_date, dte_days) for the preferred weekly expiry.
    Expiry = the (dte_prefer_hi // 7)-th Friday strictly after sim_date,
    mirroring intraday_backtest.select_expiry (2-3 week Friday expiry).
    """
    if cfg is None:
        cfg = SimConfig()
    weeks = cfg.dte_prefer_hi // 7
    d = sim_date + timedelta(days=1)
    fridays = []
    while len(fridays) < weeks:
        if d.weekday() == 4:          # Friday
            fridays.append(d)
        d += timedelta(days=1)
    expiry = fridays[-1]
    return expiry, (expiry - sim_date).days


def select_option_sim(
    ticker: str,
    direction: str,
    current_price: float,
    budget: float,
    sim_date: date,
    ticker_df: pd.DataFrame,
    cfg: SimConfig | None = None,
) -> Optional[dict]:
    """
    Pick strike + expiry + premium for a candidate trade.
    Mirrors paper_trading.select_option, with BS pricing.
    Returns None if no contract fits (rejected at any filter).
    """
    if cfg is None:
        cfg = SimConfig()
    if budget < cfg.min_trade_budget:
        return None

    is_call  = (direction == "Bullish")
    opt_type = "call" if is_call else "put"

    # ── Expiry selection: 3rd Friday strictly after sim_date (2-3 wk) ─────
    # Live agent walks an actual expiration list; in sim we synthesize the
    # date, snapped to a real weekly Friday expiry.
    expiry_dt, dte_days = _select_expiry_friday(sim_date, cfg)

    # ── IV estimate from realized vol ─────────────────────────────────────
    mask = ticker_df.index.date <= sim_date
    window = ticker_df[mask]
    if len(window) < cfg.iv_lookback_days + 1:
        return None
    iv = estimate_iv(window["Close"], cfg)

    # ── Strike candidates ─────────────────────────────────────────────────
    if is_call:
        strikes = _generate_strikes(current_price, cfg.strike_otm_pct_call_lo, cfg.strike_otm_pct_call_hi)
        if not strikes:
            strikes = _generate_strikes(current_price, 1.00, cfg.strike_fallback_call_hi)
    else:
        strikes = _generate_strikes(current_price, cfg.strike_otm_pct_put_lo, cfg.strike_otm_pct_put_hi)
        if not strikes:
            strikes = _generate_strikes(current_price, cfg.strike_fallback_put_lo, 1.001)

    if not strikes:
        return None

    # ── Score each strike: delta filter + value score ─────────────────────
    T_years = dte_days / 365.0
    spread_model = _load_spread_model(cfg) if cfg.apply_costs else None
    scored = []
    for strike in strikes:
        premium = black_scholes_price(current_price, strike, T_years,
                                       cfg.risk_free_rate, iv, opt_type)
        if premium <= 0:
            continue
        delta = black_scholes_delta(current_price, strike, T_years,
                                     cfg.risk_free_rate, iv, opt_type)
        if delta < cfg.min_delta:
            continue
        # Value score: % return on premium if stock moves assumed_move_pct
        stock_move_dollars = current_price * cfg.assumed_move_pct
        expected_gain      = delta * stock_move_dollars
        value_score        = expected_gain / premium if premium > 0 else 0.0
        # BS premium is the MID; buying crosses half the spread
        if spread_model is not None:
            sp   = spread_pct_for(strike / current_price, dte_days, spread_model)
            fill = premium * (1 + sp / 2)
        else:
            fill = premium
        cost_per = fill * 100
        if cost_per > budget:
            continue
        scored.append((value_score, strike, premium, delta, fill))

    if not scored:
        return None

    # Pick highest value score
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_strike, best_premium, best_delta, best_fill = scored[0]

    # Contracts/total_cost off the FILL; entry commission added to cost basis
    cost_per   = best_fill * 100
    contracts  = max(1, int(budget / cost_per))
    total_cost = contracts * cost_per
    if spread_model is not None:
        total_cost += cfg.commission_per_contract * contracts

    return {
        "strike":       best_strike,
        "premium":      best_fill,   # FILL when apply_costs (== mid when off)
        "contracts":    contracts,
        "total_cost":   total_cost,
        "expiry_date":  expiry_dt,
        "dte_days":     dte_days,
        "opt_type":     opt_type,
        "iv":           iv,
        "entry_delta":  best_delta,
        "value_score":  best_score,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY PIPELINE — orchestrator (one sim day)
# ═════════════════════════════════════════════════════════════════════════════

def try_market_open_entry_sim(
    state: PortfolioState,
    sim_date: date,
    spy_df: pd.DataFrame,
    ticker_data: dict,           # ticker -> DataFrame
    earnings_data: dict,         # ticker -> list[date]
    cfg: SimConfig | None = None,
    rng: random.Random | None = None,
) -> dict:
    """
    Run the full v3 entry pipeline for one sim day. Mirrors
    trading_agent_v3.try_market_open_entry.

    Side effect on success: appends an OpenTrade to state.open_trades,
    decrements state.balance, and logs to state.month_entries.
    On any skip or failure, logs to state.month_skips.

    Returns a result dict for the day's summary.
    """
    if cfg is None:
        cfg = SimConfig()
    if rng is None:
        rng = random.Random(cfg.random_seed)
    # ── Weekday gate ──────────────────────────────────────────────────────
    weekday = sim_date.weekday()
    if weekday not in cfg.entry_allowed_weekdays:
        return {
            "deployed": False, "ticker": None,
            "skip_reason": f"weekday gate ({weekday})",
            "spy_trend": None,
        }

    # ── Max open positions gate ───────────────────────────────────────────
    if len(state.open_trades) >= cfg.max_open_positions:
        return {
            "deployed": False, "ticker": None,
            "skip_reason": f"max positions ({cfg.max_open_positions}) reached",
            "spy_trend": None,
        }

    # ── Budget gate ───────────────────────────────────────────────────────
    budget = round(state.balance * cfg.base_position_pct, 2)
    if budget < cfg.min_trade_budget:
        return {
            "deployed": False, "ticker": None,
            "skip_reason": f"budget ${budget:.2f} below minimum",
            "spy_trend": None,
        }

    # ── SPY trend → direction (Neutral treated as Bullish) ────────────────
    # spy_trend is always the REAL trend (logged in month_entries even when
    # spy_mode overrides the traded direction).
    spy_trend = get_spy_trend_sim(spy_df, sim_date, cfg)
    if cfg.spy_mode == "always_bullish":
        direction = "Bullish"
    elif cfg.spy_mode == "random":
        direction = rng.choice(["Bullish", "Bearish"])
    else:  # "trend" — current behavior
        direction = "Bearish" if spy_trend == "Bearish" else "Bullish"

    # ── Scan universe ─────────────────────────────────────────────────────
    passed = []
    for ticker in cfg.ticker_universe:
        df = ticker_data.get(ticker)
        if df is None:
            state.month_skips.append({
                "date": str(sim_date), "ticker": ticker,
                "skip_reason": "no price data loaded",
            })
            continue
        result = evaluate_ticker_sim(
            ticker, direction, sim_date, df,
            earnings_data.get(ticker, []),
            cfg,
        )
        if result["passed"]:
            passed.append(result)
        else:
            state.month_skips.append({
                "date": str(sim_date), "ticker": ticker,
                "skip_reason": result["skip_reason"],
            })

    if not passed:
        return {
            "deployed": False, "ticker": None,
            "skip_reason": "no candidates passed filters",
            "spy_trend": spy_trend,
        }

    # ── Rank candidates (rank_mode changes ONLY this sort) ────────────────
    if cfg.rank_mode == "inverse_rsi":
        passed.sort(key=lambda r: r["rsi_score"])
    elif cfg.rank_mode == "random":
        rng.shuffle(passed)
    else:  # "rsi" — current behavior: RSI score descending
        passed.sort(key=lambda r: r["rsi_score"], reverse=True)

    # ── Cascade: fill up to max_open_positions slots, never opening a ─────
    # second position in a ticker already held.
    slots = min(cfg.max_open_positions - len(state.open_trades),   # >0 here, already gated
                cfg.max_new_positions_per_day)
    held  = {t.ticker for t in state.open_trades}
    deployed_tickers = []

    for candidate in passed:
        if len(deployed_tickers) >= slots:
            break
        ticker = candidate["ticker"]
        if ticker in held:
            continue

        # Recompute budget fresh for THIS fill off the decremented balance
        budget = round(state.balance * cfg.base_position_pct, 2)
        if budget < cfg.min_trade_budget:
            break   # budget exhausted, stop cascade

        option = select_option_sim(
            ticker, direction, candidate["current_price"],
            budget, sim_date, ticker_data[ticker],
            cfg,
        )
        if option is None:
            state.month_skips.append({
                "date": str(sim_date), "ticker": ticker,
                "skip_reason": "option selection failed (delta/budget/IV)",
            })
            continue

        # Deploy: create OpenTrade, decrement balance, log entry
        trade = OpenTrade(
            ticker=ticker,
            direction=direction,
            opt_type=option["opt_type"],
            strike=option["strike"],
            entry_premium=option["premium"],
            contracts=option["contracts"],
            total_cost=option["total_cost"],
            entry_date=sim_date,
            entry_price=candidate["current_price"],
            expiry_date=option["expiry_date"],
            iv=option["iv"],
            entry_delta=option["entry_delta"],
            is_volatile=(ticker in cfg.volatile_tickers),
        )
        state.open_trades.append(trade)
        state.balance -= option["total_cost"]

        state.month_entries.append({
            "date":           str(sim_date),
            "ticker":         ticker,
            "direction":      direction,
            "opt_type":       option["opt_type"],
            "strike":         option["strike"],
            "entry_premium":  round(option["premium"], 4),
            "contracts":      option["contracts"],
            "total_cost":     round(option["total_cost"], 2),
            "entry_price":    round(candidate["current_price"], 2),
            "expiry_date":    str(option["expiry_date"]),
            "iv":             round(option["iv"], 4),
            "entry_delta":    round(option["entry_delta"], 4),
            "rsi":            round(candidate["rsi"], 2),
            "value_score":    round(option["value_score"], 4),
            "spy_trend":      spy_trend,
            "is_volatile":    (ticker in cfg.volatile_tickers),
            "balance_after":  round(state.balance, 2),
        })
        held.add(ticker)
        deployed_tickers.append(ticker)

    return {
        "deployed":    bool(deployed_tickers),
        "ticker":      deployed_tickers[0] if deployed_tickers else None,
        "tickers":     deployed_tickers,
        "skip_reason": None if deployed_tickers else "all candidates exhausted",
        "spy_trend":   spy_trend,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  EXIT ENGINE — per-trade revaluation helper
# ═════════════════════════════════════════════════════════════════════════════

def revalue_trade(
    trade: OpenTrade,
    sim_date: date,
    ticker_df: pd.DataFrame,
    cfg: SimConfig | None = None,
) -> Optional[dict]:
    """
    Compute current option premium, P&L %, and underlying price for a
    single open trade as of sim_date. Returns None if pricing fails
    (e.g. missing price data on a non-trading day with no priors).

    Returns a dict:
      {
        "current_price": float,   # underlying price on sim_date
        "current_premium": float, # BS-modeled per-share premium
        "pnl_pct": float,         # P&L % vs entry_premium
        "pnl_dollar": float,      # contracts * (current - entry) * 100
        "days_remaining": int,
      }
    """
    if cfg is None:
        cfg = SimConfig()
    try:
        mask = ticker_df.index.date <= sim_date
        window = ticker_df[mask]
        if window.empty:
            return None
        current_price = float(window["Close"].iloc[-1])
    except Exception as e:
        log.warning(f"[{trade.ticker}] revalue price lookup failed: {e}")
        return None

    days_remaining = (trade.expiry_date - sim_date).days
    if days_remaining < 0:
        days_remaining = 0
    T_years = max(days_remaining / 365.0, 1.0 / 365.0)

    current_premium = black_scholes_price(
        current_price, trade.strike, T_years,
        cfg.risk_free_rate, trade.iv, trade.opt_type,
    )

    if trade.entry_premium > 0:
        pnl_pct = (current_premium - trade.entry_premium) / trade.entry_premium * 100.0
    else:
        pnl_pct = 0.0

    pnl_dollar = trade.contracts * (current_premium - trade.entry_premium) * 100.0

    return {
        "current_price":   current_price,
        "current_premium": current_premium,
        "pnl_pct":         pnl_pct,
        "pnl_dollar":      pnl_dollar,
        "days_remaining":  days_remaining,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  EXIT ENGINE — evaluate all open trades on one sim day
# ═════════════════════════════════════════════════════════════════════════════

def run_exit_engine_sim(
    state: PortfolioState,
    sim_date: date,
    ticker_data: dict,
    cfg: SimConfig | None = None,
) -> int:
    """
    Evaluate every open trade against the v3 exit rules. Rules 4 (bleed
    stop) and 5 (safety net) were removed from the sim on 2026-07-13; rule
    numbering of the remaining rules is kept to match trading_agent_v3.
    Close any that fire. Returns number of trades closed.
    """
    if cfg is None:
        cfg = SimConfig()
    if not state.open_trades:
        return 0

    closed_count = 0
    trades_to_remove = []

    for trade in state.open_trades:
        df = ticker_data.get(trade.ticker)
        if df is None:
            continue

        valuation = revalue_trade(trade, sim_date, df, cfg)
        if valuation is None:
            continue

        current_price   = valuation["current_price"]
        current_premium = valuation["current_premium"]
        pnl_pct         = valuation["pnl_pct"]
        days_remaining  = valuation["days_remaining"]

        # Update peak BEFORE evaluating trail stop
        trade.peak_pnl_pct = max(trade.peak_pnl_pct, pnl_pct)

        exit_reason = None

        # ── Rule 1: Stop loss ────────────────────────────────────────────
        if pnl_pct <= cfg.exit_stop_loss_pct:
            exit_reason = f"Stop loss ({pnl_pct:.1f}% <= {cfg.exit_stop_loss_pct:.1f}%)"

        # ── Rule 2: Hard ceiling ─────────────────────────────────────────
        elif pnl_pct >= cfg.exit_hard_ceiling_pct:
            exit_reason = f"Hard ceiling (+{pnl_pct:.1f}% >= +{cfg.exit_hard_ceiling_pct:.1f}%)"

        # ── Rule 3: Thursday-before-expiry force-exit ────────────────────
        # Fires only on the Thursday before the trade's own expiry (Friday).
        # >= (expiry - 1d) so it still fires if that Thursday is a non-trading
        # day in the daily sim. (The broad any-Thursday variant tested best on
        # 2026-05-28 with the old ~10d expiry; re-test under 2-3wk expiries.)
        elif sim_date >= (trade.expiry_date - timedelta(days=1)):
            exit_reason = "Thursday-before-expiry force-exit"

        # ── Rule 6: Trail stop ──────────────────────────────────────────
        if exit_reason is None:
            if trade.peak_pnl_pct >= cfg.exit_trail_trigger_pct:
                giveback_threshold = trade.peak_pnl_pct - cfg.exit_trail_giveback_pct
                if pnl_pct <= giveback_threshold:
                    exit_reason = (
                        f"Trail stop (peak +{trade.peak_pnl_pct:.1f}% -> "
                        f"now {pnl_pct:+.1f}%, gave back "
                        f"{trade.peak_pnl_pct - pnl_pct:.1f}pp)"
                    )

        if exit_reason is None:
            continue  # hold

        # ── Close the trade ──────────────────────────────────────────────
        # Rules above trigger on the MID premium (you can't peek at your own
        # fill); the trade is REALIZED at the fill: mid minus half the
        # spread, minus exit commission.
        if cfg.apply_costs:
            spread_model = _load_spread_model(cfg)
            sp           = spread_pct_for(trade.strike / current_price,
                                          days_remaining, spread_model)
            fill_premium = current_premium * (1 - sp / 2)
            exit_value   = (fill_premium * 100 * trade.contracts
                            - cfg.commission_per_contract * trade.contracts)
        else:
            fill_premium = current_premium
            exit_value   = current_premium * 100 * trade.contracts
        pnl_dollar = exit_value - trade.total_cost
        state.balance += exit_value

        result = "Win" if pnl_dollar > 0 else ("Loss" if pnl_dollar < 0 else "Flat")

        state.month_closed_trades.append({
            "ticker":         trade.ticker,
            "direction":      trade.direction,
            "opt_type":       trade.opt_type,
            "strike":         trade.strike,
            "entry_date":     str(trade.entry_date),
            "exit_date":      str(sim_date),
            "entry_premium":  round(trade.entry_premium, 4),
            "exit_premium":   round(fill_premium, 4),
            "contracts":      trade.contracts,
            "total_cost":     round(trade.total_cost, 2),
            "exit_value":     round(exit_value, 2),
            "pnl_pct":        round(pnl_pct, 2),
            "pnl_dollar":     round(pnl_dollar, 2),
            "peak_pnl_pct":   round(trade.peak_pnl_pct, 2),
            "result":         result,
            "exit_reason":    exit_reason,
            "days_held":      (sim_date - trade.entry_date).days,
            "days_remaining": days_remaining,
            "entry_delta":    round(trade.entry_delta, 4),
            "iv":             round(trade.iv, 4),
            "is_volatile":    trade.is_volatile,
            "balance_after":  round(state.balance, 2),
        })

        trades_to_remove.append(trade)
        closed_count += 1

    # Remove closed trades from open list
    for t in trades_to_remove:
        state.open_trades.remove(t)

    return closed_count


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — data loading and monthly flush helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data(days_back: int, cfg: SimConfig | None = None) -> tuple:
    """
    Fetch SPY + all ticker OHLCV + all ticker earnings dates upfront.
    Returns (spy_df, ticker_data, earnings_data). Tickers that fail to
    load are dropped with a warning.
    """
    if cfg is None:
        cfg = SimConfig()
    log.info(f"Loading {days_back}d of data for SPY + {len(cfg.ticker_universe)} tickers...")

    # SPY first
    spy_df = fetch_historical_data("SPY", days_back, cfg.oos_shift_days, cfg)
    if spy_df is None:
        raise RuntimeError("SPY data unavailable — cannot run sim without market trend.")

    # All ticker price data
    ticker_data = {}
    for ticker in cfg.ticker_universe:
        df = fetch_historical_data(ticker, days_back, cfg.oos_shift_days, cfg)
        if df is None or df.empty:
            log.warning(f"[{ticker}] dropped — no price data")
            continue
        ticker_data[ticker] = df

    log.info(f"Price data loaded for {len(ticker_data)}/{len(cfg.ticker_universe)} tickers")

    # Earnings dates per ticker (fail-open already inside fetch_earnings_dates)
    earnings_data = {}
    for ticker in ticker_data.keys():
        earnings_data[ticker] = fetch_earnings_dates(ticker)

    n_with_earnings = sum(1 for v in earnings_data.values() if v)
    log.info(f"Earnings data loaded for {n_with_earnings}/{len(ticker_data)} tickers "
             f"(others will fail-open on the veto)")

    return spy_df, ticker_data, earnings_data


def load_window(oos_shift_days: int, cfg: SimConfig | None = None) -> dict:
    """
    Load the full data dict for one OOS window so a sweep harness can
    cache it and reuse it across many run_sim(cfg, data=...) calls.

    Disk-cached under CACHE_DIR, keyed on (anchor, oos_shift_days,
    sim_days, tickers) so a different anchor never silently reuses the
    wrong window. cfg.use_cache=False forces a fresh yfinance fetch.
    """
    if cfg is None:
        cfg = SimConfig()
    cfg.oos_shift_days = oos_shift_days

    key = (f"{_anchor(cfg)}_{oos_shift_days}_{cfg.sim_days}_"
           f"{'-'.join(sorted(cfg.ticker_universe))}")
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")

    if cfg.use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        log.info(f"cache HIT {key}")
        return data

    spy_df, ticker_data, earnings_data = load_all_data(cfg.sim_days, cfg)
    data = {
        "spy_df":        spy_df,
        "ticker_data":   ticker_data,
        "earnings_data": earnings_data,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    log.info(f"cache MISS {key} — fetched and cached")
    return data


def flush_month(
    state: PortfolioState,
    month_idx: int,
    month_label: str,
    output_dir: str,
) -> None:
    """
    Write this month's trade activity to disk, clear per-month buffers,
    run gc.collect(). Running state (balance, open trades, balance_curve)
    is preserved.
    """
    payload = {
        "month_idx":         month_idx,
        "month_label":       month_label,
        "ending_balance":    round(state.balance, 2),
        "open_at_month_end": len(state.open_trades),
        "entries":           state.month_entries,
        "skips":             state.month_skips,
        "closed_trades":     state.month_closed_trades,
        "open_snapshot": [
            {
                "ticker":        t.ticker,
                "direction":     t.direction,
                "opt_type":      t.opt_type,
                "strike":        t.strike,
                "entry_date":    str(t.entry_date),
                "expiry_date":   str(t.expiry_date),
                "entry_premium": round(t.entry_premium, 4),
                "contracts":     t.contracts,
                "total_cost":    round(t.total_cost, 2),
                "peak_pnl_pct":  round(t.peak_pnl_pct, 2),
            }
            for t in state.open_trades
        ],
    }
    fname = f"month_{month_idx:02d}_{month_label}.json"
    fpath = os.path.join(output_dir, fname)
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    n_entries = len(state.month_entries)
    n_closed  = len(state.month_closed_trades)
    log.info(
        f"[Month {month_idx:02d} {month_label}] flushed -> {fname}  "
        f"| entries={n_entries} closed={n_closed} "
        f"balance=${state.balance:.2f} open={len(state.open_trades)}"
    )

    # Clear per-month buffers
    state.month_entries       = []
    state.month_skips         = []
    state.month_closed_trades = []
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — annual summary builder
# ═════════════════════════════════════════════════════════════════════════════

def build_annual_summary(
    output_dir: str,
    balance_curve: list,
    starting_balance: float,
    ending_balance: float,
) -> dict:
    """
    Read all month_XX_*.json files in output_dir, roll up into a single
    annual_summary.json with overall stats + caveats. Returns the summary
    dict and also writes it to disk.
    """
    # Collect all closed trades from monthly files
    all_closed = []
    all_entries = []
    monthly_summaries = []
    for fname in sorted(os.listdir(output_dir)):
        if not fname.startswith("month_") or not fname.endswith(".json"):
            continue
        with open(os.path.join(output_dir, fname), "r") as f:
            data = json.load(f)
        all_closed.extend(data.get("closed_trades", []))
        all_entries.extend(data.get("entries", []))
        monthly_summaries.append({
            "month_idx":         data.get("month_idx"),
            "month_label":       data.get("month_label"),
            "ending_balance":    data.get("ending_balance"),
            "entries":           len(data.get("entries", [])),
            "closed":            len(data.get("closed_trades", [])),
            "open_at_month_end": data.get("open_at_month_end"),
        })

    # ── Overall stats ─────────────────────────────────────────────────────
    n_trades = len(all_closed)
    wins   = [t for t in all_closed if t["pnl_dollar"] > 0]
    losses = [t for t in all_closed if t["pnl_dollar"] < 0]
    flats  = [t for t in all_closed if t["pnl_dollar"] == 0]

    win_rate = (len(wins) / n_trades * 100) if n_trades else 0.0
    avg_win  = (sum(t["pnl_dollar"] for t in wins)  / len(wins))  if wins  else 0.0
    avg_loss = (sum(t["pnl_dollar"] for t in losses) / len(losses)) if losses else 0.0
    total_pnl = sum(t["pnl_dollar"] for t in all_closed)

    # Expectancy = (win_rate * avg_win) - ((1-win_rate) * |avg_loss|)
    if n_trades:
        wr = len(wins) / n_trades
        expectancy = (wr * avg_win) + ((1 - wr) * avg_loss)
    else:
        expectancy = 0.0

    # ── Max drawdown from balance curve ──────────────────────────────────
    max_dd_pct = 0.0
    peak = starting_balance
    for _, bal in balance_curve:
        if bal > peak:
            peak = bal
        dd_pct = (bal - peak) / peak * 100 if peak > 0 else 0.0
        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct

    # ── Exit-reason breakdown ────────────────────────────────────────────
    exit_reason_counts = {}
    for t in all_closed:
        # First word of the reason string is the rule name ("Stop", "Hard",
        # "Thursday-before-expiry", "Trail")
        key = t["exit_reason"].split(" ")[0] if t.get("exit_reason") else "Unknown"
        exit_reason_counts[key] = exit_reason_counts.get(key, 0) + 1

    # ── Per-ticker breakdown ─────────────────────────────────────────────
    per_ticker = {}
    for t in all_closed:
        tk = t["ticker"]
        if tk not in per_ticker:
            per_ticker[tk] = {"trades": 0, "wins": 0, "pnl": 0.0}
        per_ticker[tk]["trades"] += 1
        if t["pnl_dollar"] > 0:
            per_ticker[tk]["wins"] += 1
        per_ticker[tk]["pnl"] += t["pnl_dollar"]
    for tk, stats in per_ticker.items():
        stats["pnl"]      = round(stats["pnl"], 2)
        stats["win_rate"] = round(stats["wins"] / stats["trades"] * 100, 1) if stats["trades"] else 0.0

    summary = {
        "config": {
            "sim_days":             SIM_DAYS,
            "starting_balance":     starting_balance,
            "ending_balance":       round(ending_balance, 2),
            "tickers":              TICKER_UNIVERSE,
            "position_pct":         BASE_POSITION_PCT,
            "max_open_positions":   MAX_OPEN_POSITIONS,
        },
        "overall": {
            "total_trades":         n_trades,
            "wins":                 len(wins),
            "losses":               len(losses),
            "flats":                len(flats),
            "win_rate_pct":         round(win_rate, 2),
            "avg_win_dollar":       round(avg_win, 2),
            "avg_loss_dollar":      round(avg_loss, 2),
            "expectancy_dollar":    round(expectancy, 2),
            "total_pnl_dollar":     round(total_pnl, 2),
            "total_return_pct":     round((ending_balance - starting_balance) / starting_balance * 100, 2),
            "max_drawdown_pct":     round(max_dd_pct, 2),
        },
        "exit_reason_counts":  exit_reason_counts,
        "per_ticker":          per_ticker,
        "monthly":             monthly_summaries,
        "balance_curve":       [(str(d), b) for d, b in balance_curve],
        "caveats": [
            "Black-Scholes premiums do not model IV crush, bid/ask spread, or the volatility smile.",
            "Earnings veto uses yfinance.earnings_dates; tickers with no historical earnings data fail open.",
            "Strike spacing in sim is $0.50; real chains are typically $1-$2.50 (mild selection bias).",
            "Target DTE is fixed at the midpoint of the preferred window (10 days), not snapped to weekly expiries.",
            "Thursday force-exit fires any time on Thursday in sim (daily granularity), not after 1pm ET.",
            "Results are useful for RELATIVE comparisons across config changes, not absolute P&L prediction.",
        ],
    }

    fpath = os.path.join(output_dir, "annual_summary.json")
    with open(fpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"Annual summary written -> {fpath}")
    return summary


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_portfolio_sim_v3()
