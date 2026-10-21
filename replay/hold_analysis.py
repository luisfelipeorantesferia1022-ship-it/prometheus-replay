"""
What if each agent trade had been held, and what was the best moment to sell?

For every trade in every agent book (or a single trades.csv) it reports:

  cost            what the trade paid per share (as booked)
  realized        what the agent actually booked
  hold_to_expiry  P&L if held to expiration: intrinsic value at the expiry
                  close (underlying close from Sharadar, unadjusted)
  peak_bid        the best price you could actually have SOLD at (bid) at any
                  minute between entry and the expiry close, when, and the
                  P&L that would have locked in
  peak_mid        the same on the mid (optimistic; nobody sells at the mid)
  ever_positive   whether the bid ever rose above the cost

Uses 1-minute option quotes (Databento cbbo-1m), entry day through expiry.

    python -m replay.hold_analysis                       # every book
    python -m replay.hold_analysis --trades path.csv --name old_sim_min1_lstm_copy
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import asof, config, dbn

OUT = config.RESULTS_DIR / "analysis"


def expiry_from_symbol(sym: str) -> date:
    return pd.to_datetime(sym[6:12], format="%y%m%d").date()


def analyze(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, t in trades.iterrows():
        if pd.isna(t.get("realized_pnl")):
            continue
        sym = t["raw_symbol"]
        exp = expiry_from_symbol(sym)
        entry_ts = pd.Timestamp(t["entry_ts"])
        n = int(t["contracts"])
        cost = float(t["option_cost"])
        is_call = str(t["opt_type"]).upper().startswith("C")

        q = dbn.option_quotes([sym], pd.Timestamp(f"{entry_ts.date()} 09:00"),
                              pd.Timestamp(f"{exp} 16:15"), purpose="analysis_quotes")
        path = pd.DataFrame()
        if not q.empty:
            path = q[(q["ts_et"] > entry_ts) & (q["ts_et"].dt.time >= pd.Timestamp("09:31").time())
                     & (q["ts_et"].dt.time <= pd.Timestamp("16:00").time())].copy()
            path = path[(path["bid"] > 0) & (path["ask"] > 0)]
            path["mid"] = (path["bid"] + path["ask"]) / 2

        S_exp = asof.unadjusted_close(t["ticker"], exp)
        K = float(t["strike"])
        intrinsic = max(0.0, (S_exp - K) if is_call else (K - S_exp)) if S_exp else None

        row = {
            "ticker": t["ticker"], "option": f"{'CALL' if is_call else 'PUT'} {K:g} exp {exp}",
            "entry": str(entry_ts), "contracts": n, "cost": round(cost, 4),
            "spent": round(cost * 100 * n, 2),
            "exit": str(t["exit_ts"]), "exit_reason": t["exit_reason"],
            "realized_pnl": round(float(t["realized_pnl"]), 2),
            "underlying_at_expiry": S_exp, "value_at_expiry": intrinsic,
            "hold_to_expiry_pnl": round((intrinsic - cost) * 100 * n, 2) if intrinsic is not None else None,
            "hold_to_expiry_pct": round((intrinsic / cost - 1) * 100, 1) if intrinsic is not None else None,
        }
        if len(path):
            pb = path.loc[path["bid"].idxmax()]
            pm = path.loc[path["mid"].idxmax()]
            row.update({
                "peak_bid": round(float(pb["bid"]), 4), "peak_bid_at": str(pb["ts_et"]),
                "peak_bid_pnl": round((float(pb["bid"]) - cost) * 100 * n, 2),
                "peak_bid_pct": round((float(pb["bid"]) / cost - 1) * 100, 1),
                "peak_mid": round(float(pm["mid"]), 4), "peak_mid_at": str(pm["ts_et"]),
                "peak_mid_pct": round((float(pm["mid"]) / cost - 1) * 100, 1),
                "ever_positive_bid": bool(path["bid"].max() > cost),
                "peak_before_exit": pd.Timestamp(pb["ts_et"]) <= pd.Timestamp(t["exit_ts"]),
                "quote_minutes": int(len(path)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = ([(a.name or Path(a.trades).stem, Path(a.trades))] if a.trades else
            [(f"{p.parent.parent.name}_{p.parent.name}", p)
             for p in sorted((config.RESULTS_DIR / "agent").glob("*/*/trades.csv"))])
    for name, p in jobs:
        df = analyze(pd.read_csv(p))
        df.to_csv(OUT / f"hold_{name}.csv", index=False)
        if df.empty:
            continue
        pos = int(df.get("ever_positive_bid", pd.Series(dtype=bool)).fillna(False).sum())
        held_win = int((df["hold_to_expiry_pnl"] > 0).sum())
        print(f"::notice title=Hold analysis {name}::{len(df)} trades | booked "
              f"${df.realized_pnl.sum():,.0f} | held to expiry ${df.hold_to_expiry_pnl.sum():,.0f} "
              f"({held_win} winners) | bid ever above cost: {pos} | sum of peak-bid P&L "
              f"${df.get('peak_bid_pnl', pd.Series(dtype=float)).sum():,.0f}", flush=True)
    print(f"Databento spend now ${dbn.ledger()['spent_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
