"""
Stages 2-4 for every week, plus the Databento prefetch for stage 2.

    python -m replay.weekly prefetch            # buys 1-minute bars stage 2 needs
    DATABENTO_QUOTE_ONLY=1 python -m replay.weekly prefetch   # price it only
    python -m replay.weekly run                 # stages 2, 3, 4 for all weeks

Stage 2 is intraday_backtest.run_intraday_backtest(), unmodified, with its
data sources rebound: daily bars from Sharadar as of the weekend, and the
15-minute bars yfinance's `period="60d"` would have returned that weekend,
built from Databento 1-minute bars.

Stage 2 -> 3 gate (requested 2026-10-16, differs from the live console): a
candidate's total stage-2 P&L must be at least STAGE2_MIN_GAIN_PCT (200%) of
the fixed $10,000 per-trade stake.

Stage 3 is portfolio_sim_v3.run_sim(), unmodified, fed the same point-in-time
data through its `data=` argument. It runs twice: on the gate survivors, then
again on only the tickers that finished positive in the first sim (removing the
losers changes who gets the three slots).

Stage 4 promotes the tickers positive in the second sim. When nothing is
promoted the whitelist is left as it was, exactly as the console does; the
replay assumes Apply was pressed every weekend.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import pandas as pd

from . import asof, config, dbn, promotion, shims
from .stage1 import out_dir

INTRADAY_LOOKBACK_DAYS = 60   # yfinance period="60d"


def weekend_anchor(asof_date: date) -> date:
    """The Saturday the weekly run happens; the strategy code's 'today'."""
    d = asof_date + timedelta(days=1)
    while d.weekday() != 5:
        d += timedelta(days=1)
    return d


def candidates_for(asof_date: date) -> list[str]:
    p = out_dir(asof_date) / "stage1.json"
    if not p.exists():
        raise FileNotFoundError(f"stage 1 not merged for {asof_date}")
    return json.loads(p.read_text())["candidates"]


def intraday_days(asof_date: date) -> list[date]:
    anchor = weekend_anchor(asof_date)
    return asof.trading_days(anchor - timedelta(days=INTRADAY_LOOKBACK_DAYS), asof_date)


# ─────────────────────────────────────────────────────────────────────────────
#  Prefetch
# ─────────────────────────────────────────────────────────────────────────────

def plan() -> dict[date, set[str]]:
    need: dict[date, set[str]] = {}
    for a in asof.weekly_asof_dates():
        cands = candidates_for(a)
        for d in intraday_days(a):
            need.setdefault(d, set()).update(cands)
    return need


def prefetch() -> None:
    need = plan()
    pairs = sum(len(v) for v in need.values())
    print(f"[weekly] stage-2 bars: {pairs} ticker-days over {len(need)} days", flush=True)
    before = dbn.ledger()
    for i, d in enumerate(sorted(need), 1):
        dbn.prefetch_equity_day(d, sorted(need[d]), purpose="stage2_bars")
        if i % 25 == 0:
            led = dbn.ledger()
            print(f"  {i}/{len(need)} days | spent ${led['spent_usd']:.2f} "
                  f"quoted ${led.get('quoted_usd', 0):.2f}", flush=True)
    after = dbn.ledger()
    print(f"[weekly] prefetch done: spent ${after['spent_usd'] - before['spent_usd']:.2f}, "
          f"quoted ${after.get('quoted_usd', 0) - before.get('quoted_usd', 0):.2f}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 data shim
# ─────────────────────────────────────────────────────────────────────────────

def intraday_15m(ticker: str, asof_date: date) -> pd.DataFrame | None:
    frames = []
    for d in intraday_days(asof_date):
        b = dbn.equity_bars(ticker, d, purpose="stage2_bars")
        if b.empty:
            continue
        b = b.between_time("09:30", "15:59")
        if b.empty:
            continue
        r = b.resample("15min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        frames.append(r)
    if not frames:
        return None
    df = pd.concat(frames).rename(columns=str.capitalize)
    df["session"] = df.index.normalize()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  One week
# ─────────────────────────────────────────────────────────────────────────────

def run_week(asof_date: date, prev_whitelist: list[str]) -> dict:
    import intraday_backtest as ib
    from portfolio_sim_v3 import SimConfig, run_sim

    od = out_dir(asof_date)
    anchor = weekend_anchor(asof_date)
    cands = candidates_for(asof_date)
    print(f"\n[weekly] {asof_date} (run {anchor}): {len(cands)} candidates", flush=True)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    bt_df = None
    if cands:
        ib.TICKERS = list(cands)
        ib.CFG = SimConfig(ticker_universe=list(cands), oos_shift_days=0,
                           anchor_date=anchor.isoformat(), use_cache=False)
        ib.fetch_historical_data = shims.sim_fetch(anchor)
        ib.fetch_intraday_data = lambda t: intraday_15m(t, asof_date)
        ib.fetch_earnings_dates = asof.earnings_dates
        ib.correlate_with_lstm_metrics = lambda df: None
        ib.OUTPUT_CSV = str(od / "intraday_backtest_results.csv")
        p = od / "intraday_backtest_results.csv"
        p.unlink(missing_ok=True)
        ib.run_intraday_backtest()
        bt_df = pd.read_csv(p) if p.exists() else None

    # ── Gate: stage 2 -> 3 ────────────────────────────────────────────────
    stake = float(getattr(ib, "BUDGET_PER_TRADE", 10000.0))
    gate_df, gated = promotion.stage2_gate(cands, bt_df, stake, config.STAGE2_MIN_GAIN_PCT,
                                           config.MIN_TRADES)

    # ── Stage 3: sim 1 on gate survivors, sim 2 on sim-1 positives ────────
    def _sim(universe: list[str], tag: str):
        if not universe:
            return None
        cfg = SimConfig(ticker_universe=list(universe), oos_shift_days=0,
                        anchor_date=anchor.isoformat(),
                        max_open_positions=config.MAX_OPEN_SIM, use_cache=False)
        m = run_sim(cfg, data=shims.sim_data(universe, anchor), write_output=False)
        if not m:
            return None
        pd.DataFrame(m.pop("closed_trades", [])).to_csv(od / f"{tag}_closed_trades.csv", index=False)
        eq = m.pop("equity_curve", [])
        pd.DataFrame(eq, columns=["date", "equity"]).to_csv(od / f"{tag}_equity.csv", index=False)
        (od / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2, default=str))
        return m

    m1 = _sim(gated, "sim1")
    sim1, sim1_pos = promotion.sim_survivors(gated, m1, config.MIN_TRADES)
    m2 = _sim(sim1_pos, "sim2")
    sim2, sim2_pos = promotion.sim_survivors(sim1_pos, m2, config.MIN_TRADES)

    # ── Stage 4 ───────────────────────────────────────────────────────────
    promoted = list(sim2_pos)
    table = promotion.funnel_table(gate_df, sim1, sim2, config.STAGE2_MIN_GAIN_PCT)
    table.to_csv(od / "promotion_table.csv", index=False)
    applied = promoted if promoted else list(prev_whitelist)
    summary = {
        "asof": asof_date.isoformat(), "run_date": anchor.isoformat(),
        "stage2_min_gain_pct": config.STAGE2_MIN_GAIN_PCT,
        "candidates": cands, "passed_stage2_gate": gated,
        "sim1_positive": sim1_pos, "promoted": promoted, "whitelist": applied,
        "carried_over": not promoted,
        "backtest_trades": int(len(bt_df)) if bt_df is not None else 0,
        "sim1_trades": int(m1["n_trades"]) if m1 else 0,
        "sim2_trades": int(m2["n_trades"]) if m2 else 0,
    }
    (od / "weekly.json").write_text(json.dumps(summary, indent=2))
    print(f"[weekly] {asof_date}: {len(cands)} candidates -> {len(gated)} pass stage-2 gate -> "
          f"{len(sim1_pos)} positive in sim 1 -> {len(promoted)} positive in sim 2 "
          f"-> whitelist {applied}", flush=True)
    return summary


def run_all() -> None:
    whitelist: list[str] = []
    rows = []
    for a in asof.weekly_asof_dates():
        s = run_week(a, whitelist)
        whitelist = s["whitelist"]
        rows.append({"asof": s["asof"], "n_candidates": len(s["candidates"]),
                     "n_passed_stage2_gate": len(s["passed_stage2_gate"]),
                     "n_sim1_positive": len(s["sim1_positive"]),
                     "promoted": " ".join(s["promoted"]),
                     "whitelist": " ".join(s["whitelist"]),
                     "carried_over": s["carried_over"]})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "whitelist_timeline.csv", index=False)


def whitelist_timeline() -> dict[date, list[str]]:
    df = pd.read_csv(config.RESULTS_DIR / "whitelist_timeline.csv")
    return {date.fromisoformat(r.asof): (str(r.whitelist).split() if isinstance(r.whitelist, str) else [])
            for r in df.itertuples()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prefetch", "run"])
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(config.STRATEGY_DIR))
    if a.cmd == "prefetch":
        prefetch()
    else:
        run_all()


if __name__ == "__main__":
    main()
