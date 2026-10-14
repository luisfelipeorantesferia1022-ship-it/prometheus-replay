"""
intraday_backtest.py — Peak-catcher exit test for Prometheus

PURPOSE
-------
Daily backtests re-price each option ONLY on daily closes, so any intraday
spike that reverts before 4pm is invisible to them. This script tests one
hypothesis: does a continuously-monitoring intraday exit ("peak catcher") have
an edge that the end-of-day exit misses?

ENTRY SIDE — v3 (changed)
-------------------------
This file previously imported its entire entry side from V2/backtest.py, which
chose direction from an LSTM signal and bought a strike fixed at +/-3%. The
live agent does neither. That made this harness and the portfolio sim score
different trades on the same tickers, so promoting on "positive in both" was
intersecting two unrelated strategies.

Entries now come from portfolio_sim_v3 — the same functions the sim runs:
SPY's 20-day trend sets direction (Neutral treated as Bullish), earnings
within the hold window veto the trade, and the strike is chosen by delta
(>= 0.35) within budget rather than at a fixed offset.

What deliberately does NOT carry over is portfolio state. Stage 2 prices every
candidate trade independently — no running balance, no max-open cap, no
ranking — so the exit-model comparison is not confounded by position limits or
compounding. Stage 3 is where the book is simulated. Read per-ticker P&L here
as "how did this ticker's trades resolve under each exit model", not as a
portfolio return.

It swaps ONLY the exit simulation for an intraday premium-path walk, and
reports three exit models side by side:

  oracle_peak : sell at the best premium reached at ANY intraday bar.
                Theoretical ceiling. If this is negative, the thesis is dead.
  trailing    : the v3 exit stack applied to every intraday bar —
                -75% stop, +200% ceiling, +50% trail / 20pp giveback,
                Thursday-before-expiry 1pm force exit. This is the live engine.
  eod_close   : the SAME v3 stack sampled only at session closes — the control
                that isolates the value of intraday monitoring vs daily checks.

The headline metric is "rescued": trades that LOST or EXPIRED under eod_close
but finished green under trailing. That number is the thesis, quantified.

DATA CONSTRAINT
---------------
yfinance intraday history is capped: 1m ~= 7 days, 5m/15m ~= 60 calendar days.
Holding window now runs to the Thursday-before-expiry of a 3-week option, so
the most recent ~3 weeks of entries will have their intraday path TRUNCATED at
the last available bar (their expiry is still in the future / past the data
cap). Those resolve via the time-stop fallback, not a real Thursday exit. For
full 3-week holds, swap fetch_intraday_data() for a Tradier /markets/timesales call.

LIMITATIONS (v1 — be aware before trusting numbers)
  - IV is held constant at entry IV across the whole hold (no intraday IV
    change / no vol crush). Same simplification as backtest.py.
  - Entry fill assumed at the signal day's close; intraday monitoring begins
    the next session. No slippage / commission modelled.
  - Delta is not used for the intraday path — the option is re-priced directly
    from the intraday underlying via Black-Scholes, which is more accurate than
    a delta approximation.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import math
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr, spearmanr

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Entry side comes from v3, not v2 ─────────────────────────────────────────
# This file used to import its whole entry side from V2/backtest.py, which
# picked direction from an LSTM signal and bought a strike fixed at +/-3%.
# The live agent does neither: SPY's 20-day trend sets direction, and the
# strike is chosen by delta. Stage 2 and stage 3 were therefore scoring
# different trades on the same tickers, and promotion was intersecting two
# unrelated strategies.
#
# Everything below now comes from portfolio_sim_v3 — the same functions the
# portfolio sim runs — so both harnesses enter on the same rule. The intraday
# exit simulation further down is unchanged; it remains the point of this file.
from portfolio_sim_v3 import (
    SimConfig,
    black_scholes_price,     # used by the intraday exit sim below
    fetch_historical_data,
    fetch_earnings_dates,
    get_spy_trend_sim,
    evaluate_ticker_sim,     # earnings veto + RSI + price
    select_option_sim,       # strike/expiry/premium by delta and budget
)

# Injectable config. run_backtest.py rebinds this so stage 2 and stage 3 share
# one set of entry parameters (delta floor, allowed weekdays, expiry window,
# earnings veto). Left as None, a default SimConfig is used.
CFG: "SimConfig | None" = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("IntradayBT")

# ─────────────────────────────────────────────────────────────────────────────
#  2. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = [
    'ATO', 'AFL', 'UNH', 'UBER', 'BA', 'DLTR',
    'ETN', 'TSCO', 'COST', 'MDT', 'BNY', 'CPRT',
    'GPN', 'NVR', 'KHC', 'DLR', 'CARR',
]

# ── Entry gauntlet (mirrors portfolio_sim_v3 / trading_agent_v3) ─────────────
# The gate itself now lives in portfolio_sim_v3 — direction from SPY's 20-day
# trend, earnings veto, RSI, delta >= 0.35 strike selection. Only the two
# numbers this file owns are set here.
#
# Note BUDGET_PER_TRADE is a FIXED notional, not a percentage of a running
# balance. Stage 2 deliberately has no portfolio state: it prices every
# candidate trade independently so the exit-model comparison is not confounded
# by position limits or compounding. Stage 3 is where the book is simulated.
BACKTEST_DAYS    = 30        # how many trading days of entries to scan
BUDGET_PER_TRADE = 10000.0   # fixed notional per trade
RISK_FREE        = 0.045

# ── DTE selection — 1/2/3-week weeklies, 3 weeks (21d) PREFERRED ─────────────
# The backtest always uses the preferred expiry because Black-Scholes has no
# liquidity/chain constraint — the 1wk/2wk fallbacks only matter in live
# trading when a 3wk contract is unavailable. Expiry is the 3rd Friday after
# entry; DTE is computed exactly from the calendar.
DTE_PREFERRED_WEEKS = 3
DTE_ALLOWED_WEEKS   = (1, 2, 3)   # documentation only (see note above)

# ── Exit rules — MATCHED to trading_agent_v3.py so this predicts v3 ──────────
# Values you named: stop -75, ceiling +200, Thursday-before-expiry force exit.
# Values taken from v3 (you did NOT name these — override if wrong):
#   trail trigger +50%, giveback 20pp.
EXIT_STOP_LOSS_PCT      = -75.0   # hard stop  (YOU SET THIS)
EXIT_CEILING_PCT        = 200.0   # hard ceiling, immediate exit  (YOU SET THIS)
EXIT_TRAIL_TRIGGER_PCT  = 50.0    # arm trailing once P&L >= this  (from v3)
EXIT_TRAIL_GIVEBACK_PCT = 20.0    # exit if P&L drops this many pts from peak (from v3)
FORCE_EXIT_HOUR_ET      = 13      # Thursday-before-expiry, 1pm ET  (YOU SET THIS)

INTRADAY_INTERVAL  = "15m"   # yfinance: "5m" or "15m" stay inside the 60-day window
OUTPUT_CSV         = "intraday_backtest_results.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  3. INTRADAY DATA FETCHER  (swap this for Tradier timesales for longer history)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_intraday_data(ticker: str) -> pd.DataFrame | None:
    """
    Download intraday OHLC bars for a ticker at INTRADAY_INTERVAL.
    yfinance caps history (5m/15m ~ 60 calendar days), so this only covers
    the recent window — which is why the holding period must stay short.

    Returns a tz-naive DataFrame indexed by timestamp with a Close column,
    or None if nothing usable came back.
    """
    try:
        df = yf.download(
            ticker,
            period="60d",
            interval=INTRADAY_INTERVAL,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            log.warning(f"[{ticker}] No intraday data returned.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df[["Open", "High", "Low", "Close"]].dropna()
        # Normalise timezone so we can compare against daily .date() cleanly
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        df["session"] = df.index.normalize()
        log.info(f"[{ticker}] Loaded {len(df)} intraday bars "
                 f"({df.index[0]} -> {df.index[-1]})")
        return df
    except Exception as e:
        log.error(f"[{ticker}] fetch_intraday_data failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  4a. EXPIRY SELECTION  (1/2/3-week weeklies, 3 weeks preferred)
# ─────────────────────────────────────────────────────────────────────────────
def select_expiry(entry_date):
    """
    Return (expiry_friday_date, dte_days) for the preferred weekly expiry.
    Expiry = the DTE_PREFERRED_WEEKS-th Friday strictly after entry_date.
    """
    d = entry_date + timedelta(days=1)
    fridays = []
    while len(fridays) < DTE_PREFERRED_WEEKS:
        if d.weekday() == 4:          # Friday
            fridays.append(d)
        d += timedelta(days=1)
    expiry = fridays[-1]
    return expiry, (expiry - entry_date).days


# ─────────────────────────────────────────────────────────────────────────────
#  4b. INTRADAY EXIT SIMULATION  (the only logic that differs from backtest.py)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_exits(entry_dt, entry_premium, strike, opt_type, iv,
                   dte_days, expiry_date, intraday_df: pd.DataFrame) -> dict | None:
    """
    Walk the intraday premium path from the session after entry up to the
    Thursday-before-expiry (1pm ET force exit) and resolve three exit models
    against the SAME path, using the v3 exit stack:
        hard stop -75%  ->  hard ceiling +200%  ->  +50% trail / 20pp giveback
        ->  Thursday-before-expiry 1pm force exit  ->  time-stop if data runs out.

    Models differ ONLY in sampling frequency, isolating the value of intraday
    monitoring vs once-a-day checking:
        oracle_peak : best premium reachable on the path (theoretical ceiling)
        trailing    : v3 rules applied to EVERY intraday bar (the real engine)
        eod_close   : v3 rules applied only at session closes (daily resolution)

    Returns a result dict, or None if no intraday bars cover the hold window
    (e.g. entry predates yfinance's ~60-day intraday cap).
    """
    entry_date = pd.Timestamp(entry_dt).normalize()
    thursday   = pd.Timestamp(expiry_date) - pd.Timedelta(days=1)   # Thu before Fri expiry
    force_cutoff = thursday.normalize() + pd.Timedelta(hours=FORCE_EXIT_HOUR_ET)

    # Sessions after entry, up to and including the Thursday-before-expiry.
    future = intraday_df[(intraday_df["session"] > entry_date) &
                         (intraday_df["session"] <= thursday.normalize())]
    if future.empty:
        return None

    def premium_at(ts, s_price):
        elapsed = max((pd.Timestamp(ts).normalize() - entry_date).days, 0)
        remaining_T = max((dte_days - elapsed) / 365.0, 1.0 / 365.0)
        return black_scholes_price(s_price, strike, remaining_T, RISK_FREE, iv, opt_type)

    def pct(prem):
        return (prem - entry_premium) / entry_premium * 100.0

    # Shared exit rule stack — identical for both models; only the sampled
    # sequence of (timestamp, premium, pnl%) differs.
    def run_rules(seq):
        armed = False
        peak_p = -1e9
        for ts, prem, p in seq:
            if p <= EXIT_STOP_LOSS_PCT:                 # 1. hard stop
                return p, "Loss"
            if p >= EXIT_CEILING_PCT:                   # 2. hard ceiling
                return p, "Win"
            if not armed and p >= EXIT_TRAIL_TRIGGER_PCT:
                armed = True
            if armed:                                   # 3. trailing giveback
                peak_p = max(peak_p, p)
                if p <= peak_p - EXIT_TRAIL_GIVEBACK_PCT:
                    return p, "Win" if p > 0 else "Loss"
            if ts >= force_cutoff:                      # 4. Thu-before-expiry force
                return p, "Win" if p > 0 else ("Loss" if p < 0 else "Flat")
        # data ran out before Thursday -> time-stop at last available bar
        last_p = seq[-1][2]
        return last_p, "Win" if last_p > 0 else ("Loss" if last_p < 0 else "Flat")

    # ── Build the full intraday premium series ───────────────────────────────
    bars = []
    for ts, row in future.iterrows():
        prem = premium_at(ts, float(row["Close"]))
        bars.append((ts, prem, pct(prem)))
    if not bars:
        return None

    # ── Model A: oracle_peak — best premium reachable on the path ────────────
    peak_idx = max(range(len(bars)), key=lambda i: bars[i][1])
    oracle_ts, _, oracle_pct = bars[peak_idx]

    # ── Model B: trailing — v3 stack on every intraday bar ───────────────────
    tr_pct, tr_result = run_rules(bars)

    # ── Model C: eod_close — v3 stack sampled only at session closes ─────────
    eod_bars = []
    for sess in sorted(future["session"].unique()):
        day = future[future["session"] == sess]
        ts  = day.index[-1]
        prem = premium_at(ts, float(day.iloc[-1]["Close"]))
        eod_bars.append((ts, prem, pct(prem)))
    eod_pct, eod_result = run_rules(eod_bars)

    return {
        "dte_days":     dte_days,
        "expiry":       str(pd.Timestamp(expiry_date).date()),
        "oracle_pct":   round(oracle_pct, 2),
        "oracle_when":  str(oracle_ts),
        "trailing_pct": round(tr_pct, 2),
        "trailing_res": tr_result,
        "eod_pct":      round(eod_pct, 2),
        "eod_res":      eod_result,
        "n_bars":       len(bars),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_intraday_backtest() -> None:
    all_trades = []
    cfg = CFG if CFG is not None else SimConfig()

    # Daily history must cover the entry window plus warmup for RSI and IV.
    days_back = max(BACKTEST_DAYS * 2 + 120, 260)

    log.info("Fetching SPY daily data for the trend filter...")
    spy_df = fetch_historical_data("SPY", days_back, cfg=cfg)
    if spy_df is None:
        log.warning("SPY data unavailable — every day will read as Neutral "
                    "(treated as Bullish, exactly as the live agent does).")

    for ticker in TICKERS:
        log.info("=" * 50)
        log.info(f"  {ticker} — fetching daily + intraday data ...")

        df = fetch_historical_data(ticker, days_back, cfg=cfg)
        if df is None or df.empty:
            log.warning(f"  [{ticker}] Skipped — no daily data.")
            continue
        if len(df) < cfg.entry_rsi_period + BACKTEST_DAYS + 5:
            log.warning(f"  [{ticker}] Skipped — insufficient daily rows.")
            continue

        intraday_df = fetch_intraday_data(ticker)
        if intraday_df is None:
            log.warning(f"  [{ticker}] Skipped — no intraday data.")
            continue

        # Earnings veto data. fetch_earnings_dates fails open (returns []),
        # so a lookup failure never silently blocks every entry.
        try:
            earnings = fetch_earnings_dates(ticker)
        except Exception:
            earnings = []

        # The last BACKTEST_DAYS trading days present in the daily frame.
        sim_dates = [d.date() if hasattr(d, "date") else d
                     for d in df.index[-BACKTEST_DAYS:]]

        n_trades = n_no_intraday = 0
        n_weekday = n_gate = n_option = 0

        for sim_date in sim_dates:
            # ── v3 entry gate, in the live agent's order ─────────────────
            # 1. Weekday. The agent only opens Mon/Tue/Wed.
            if sim_date.weekday() not in cfg.entry_allowed_weekdays:
                n_weekday += 1
                continue

            # 2. SPY trend sets direction. Neutral is treated as Bullish,
            #    matching try_market_open_entry_sim's "trend" mode.
            spy_trend = get_spy_trend_sim(spy_df, sim_date, cfg) \
                        if spy_df is not None else "Neutral"
            direction = "Bearish" if spy_trend == "Bearish" else "Bullish"

            # 3. Earnings veto + RSI + price, straight from the sim.
            ev = evaluate_ticker_sim(ticker, direction, sim_date, df,
                                     earnings, cfg)
            if not ev.get("passed"):
                n_gate += 1
                continue

            current_price = float(ev["current_price"])

            # 4. Strike/expiry/premium by delta and budget — not a fixed ±3%.
            option = select_option_sim(ticker, direction, current_price,
                                       BUDGET_PER_TRADE, sim_date, df, cfg)
            if option is None:
                n_option += 1
                continue

            opt_type      = option["opt_type"]
            strike        = option["strike"]
            entry_premium = option["premium"]
            iv            = option["iv"]
            dte_days      = option["dte_days"]
            expiry_date   = option["expiry_date"]
            contracts     = option["contracts"]
            total_cost    = option["total_cost"]
            entry_date    = sim_date
            snr           = float(ev.get("rsi_score", 0.0))

            # ── INTRADAY EXIT SIM (the differentiator) ───────────────────────
            res = simulate_exits(entry_date, entry_premium, strike, opt_type, iv,
                                  dte_days, expiry_date, intraday_df)
            if res is None:
                n_no_intraday += 1
                continue

            all_trades.append({
                "ticker":        ticker,
                "date":          str(pd.Timestamp(entry_date).date()),
                "direction":     direction,
                "option_type":   opt_type.upper(),
                "strike":        round(strike, 2),
                "dte_days":      res["dte_days"],
                "expiry":        res["expiry"],
                "entry_premium": round(entry_premium, 4),
                "contracts":     contracts,
                "total_cost":    round(total_cost, 2),
                "entry_delta":   round(float(option["entry_delta"]), 4),
                "rsi":           round(float(ev.get("rsi", 0.0)), 2),
                "rsi_score":     round(float(ev.get("rsi_score", 0.0)), 2),
                "spy_trend":     spy_trend,
                "snr":           round(snr, 4),
                "oracle_pct":    res["oracle_pct"],
                "trailing_pct":  res["trailing_pct"],
                "trailing_res":  res["trailing_res"],
                "eod_pct":       res["eod_pct"],
                "eod_res":       res["eod_res"],
                "oracle_when":   res["oracle_when"],
                "n_bars":        res["n_bars"],
            })
            n_trades += 1

        log.info(f"  [{ticker}] trades={n_trades}  no_intraday={n_no_intraday}  "
                 f"skipped: weekday={n_weekday} gate={n_gate} option={n_option}")

    report(all_trades)


# ─────────────────────────────────────────────────────────────────────────────
#  6. REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _model_expectancy(df_trades, pct_col, total_col="total_cost"):
    """Dollar P&L and per-trade expectancy for one exit model."""
    pnl_dollar = df_trades[total_col] * df_trades[pct_col] / 100.0
    return pnl_dollar.sum(), df_trades[pct_col].mean()


def _find_lstm_csv():
    """Locate lstm_accuracy_results.csv in CWD, ./V2, or beside this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in ["lstm_accuracy_results.csv",
                 os.path.join("V2", "lstm_accuracy_results.csv"),
                 os.path.join(here, "lstm_accuracy_results.csv"),
                 os.path.join(here, "V2", "lstm_accuracy_results.csv")]:
        if os.path.exists(path):
            return path
    return None


def correlate_with_lstm_metrics(df: pd.DataFrame) -> None:
    """
    Join per-ticker intraday outcomes with LSTM qualification metrics and print
    rank (Spearman) + linear (Pearson) correlations. Exploratory only: small n,
    noisy per-ticker P&L. Spearman is the one to trust (robust to outlier names).
    """
    lstm_path = _find_lstm_csv()
    if not lstm_path:
        print("\n(correlation skipped — lstm_accuracy_results.csv not found)")
        return
    lstm = pd.read_csv(lstm_path)

    METRICS  = ["dir_accuracy_pct", "hit_rate_pct", "avg_mae_pct", "avg_snr"]
    OUTCOMES = ["trailing_total_$", "trailing_avg_%", "trailing_winrate_%",
                "oracle_avg_%", "eod_avg_%"]

    rows = []
    for tk, sub in df.groupby("ticker"):
        rows.append({
            "ticker":             tk,
            "n_trades":           len(sub),
            "trailing_total_$":   (sub["total_cost"] * sub["trailing_pct"] / 100).sum(),
            "trailing_avg_%":     sub["trailing_pct"].mean(),
            "trailing_winrate_%": (sub["trailing_res"] == "Win").mean() * 100,
            "oracle_avg_%":       sub["oracle_pct"].mean(),
            "eod_avg_%":          sub["eod_pct"].mean(),
        })
    perf = pd.DataFrame(rows)
    merged = perf.merge(lstm[["ticker"] + METRICS], on="ticker", how="inner")
    merged = merged.sort_values("trailing_total_$", ascending=False)

    print("\n" + "=" * 78)
    print("  LSTM METRICS vs INTRADAY P&L  (per ticker that traded)")
    print("=" * 78)
    show = merged[["ticker", "n_trades", "dir_accuracy_pct", "hit_rate_pct",
                   "avg_mae_pct", "avg_snr", "trailing_avg_%",
                   "trailing_winrate_%", "trailing_total_$", "oracle_avg_%"]]
    print(show.to_string(index=False,
          formatters={"trailing_total_$": lambda v: f"{v:+,.0f}"}))
    n = len(merged)
    print(f"\nTickers in correlation: {n}  ({len(lstm) - n} never traded / not in run)")
    if n < 5:
        print("Too few traded tickers for meaningful correlation.")
        return

    print("\n" + "-" * 78)
    print("  Spearman rho (rank) / Pearson r (linear) with p-values.")
    print("  Trust Spearman. A SINGLE flagged cell at this n is probably noise;")
    print("  look for the same metric repeating its sign across outcomes.")
    print("-" * 78)
    for outcome in OUTCOMES:
        print(f"\n  Outcome: {outcome}")
        print(f"    {'metric':<18}{'Spearman':>11}{'p':>8}{'Pearson':>10}{'p':>8}")
        print("    " + "-" * 54)
        y = merged[outcome].astype(float)
        for m in METRICS:
            x = merged[m].astype(float)
            try:
                rho, p_s = spearmanr(x, y)
                r,   p_p = pearsonr(x, y)
                flag = "  <--" if (abs(rho) > 0.5 and p_s < 0.05) else ""
                print(f"    {m:<18}{rho:>+11.2f}{p_s:>8.3f}{r:>+10.2f}{p_p:>8.3f}{flag}")
            except Exception as e:
                print(f"    {m:<18}  (failed: {e})")
    print("=" * 78)


def report(all_trades: list) -> None:
    print()
    print("=" * 60)
    print("  INTRADAY PEAK-CATCHER BACKTEST")
    print("=" * 60)

    if not all_trades:
        print("No trades were recorded (check intraday data availability).")
        return

    df = pd.DataFrame(all_trades)
    n = len(df)

    # Per-model aggregate
    print(f"Total trades: {n}\n")
    print(f"{'MODEL':<14}{'TotalP&L$':>14}{'AvgP&L%':>10}{'Win%':>8}")
    print("-" * 46)
    for label, pct_col, res_col in [
        ("eod_close",   "eod_pct",      "eod_res"),
        ("trailing",    "trailing_pct", "trailing_res"),
        ("oracle_peak", "oracle_pct",   None),
    ]:
        tot, avg = _model_expectancy(df, pct_col)
        if res_col:
            winpct = (df[res_col] == "Win").sum() / n * 100
        else:
            winpct = (df[pct_col] > 0).sum() / n * 100
        print(f"{label:<14}{tot:>+14.0f}{avg:>+10.1f}{winpct:>7.0f}%")

    # ── THE THESIS METRIC: trades rescued by intraday exit ───────────────────
    eod_bad   = df["eod_res"].isin(["Loss", "Flat"])
    rescued   = df[eod_bad & (df["trailing_res"] == "Win")]
    sacrificed = df[(df["eod_res"] == "Win") & (df["trailing_res"] != "Win")]

    print()
    print("-" * 46)
    print(f"Rescued by peak-catcher (eod lost -> trailing won): {len(rescued)}")
    print(f"Sacrificed (eod won -> trailing did not):           {len(sacrificed)}")
    if len(rescued):
        gain = (rescued["total_cost"] * rescued["trailing_pct"] / 100).sum()
        print(f"  Net $ from rescued trades (trailing):             ${gain:+,.0f}")

    # ── Oracle headroom: money intraday leaves on the table vs eod ───────────
    headroom = ((df["oracle_pct"] - df["eod_pct"]) > 5).sum()
    print(f"Trades where oracle peak beat eod by >5pts:         {headroom} / {n}")
    print("  (high count => intraday spikes exist to chase; "
          "low count => no edge to capture)")

    # Per-ticker trailing breakdown
    print()
    print("--- Per-ticker (trailing model) ---")
    print(f"{'TICK':<6}{'N':>4}{'Win':>5}{'AvgP&L%':>10}{'Total$':>12}")
    print("-" * 37)
    for t in TICKERS:
        sub = df[df["ticker"] == t]
        if sub.empty:
            print(f"{t:<6}{'0':>4}{'—':>5}{'—':>10}{'—':>12}")
            continue
        w = (sub["trailing_res"] == "Win").sum()
        tot, avg = _model_expectancy(sub, "trailing_pct")
        print(f"{t:<6}{len(sub):>4}{w:>5}{avg:>+10.1f}{tot:>+12.0f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull trade log written to {OUTPUT_CSV}")
    print("=" * 60)

    correlate_with_lstm_metrics(df)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_intraday_backtest()
