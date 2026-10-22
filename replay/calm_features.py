"""
More entry-time features for every option in the calm-stock test, all from
Sharadar daily data (no Databento cost), so winners and losers can be searched
for things knowable at 9:46 on the entry morning.

Per stock (as of the prior close unless noted):
  gap at the open, move from the open to 9:46, 1/5/20/60-day returns,
  distance from the 20- and 50-day averages, RSI(14), distance from the
  52-week high and low, yesterday's range and volume vs its 20-day average,
  ATR(14), days to the next earnings report and whether it falls before the
  Thursday exit, days since the last one.
Market: SPY vs its 20-day average, SPY 5/20-day return, SPY gap, SPY 20-day vol.
Option: implied vol / 20-day realised vol, premium in dollars and as % of the
  stock, how far out of the money.
Direction-bearing features also get a "_with" copy, signed so that positive
means the feature points the way of the option (up for a call, down for a put).

    python -m replay.calm_features      # results/analysis/calm_features.csv
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from . import asof, config

R = config.RESULTS_DIR / "analysis"
SIGNED = ["gap_pct", "early_move_pct", "ret_1d_pct", "ret_5d_pct", "ret_20d_pct", "ret_60d_pct",
          "dist_ma20_pct", "dist_ma50_pct", "rsi_minus_50", "dist_52w_high_pct", "dist_52w_low_pct",
          "spy_vs_ma20_pct", "spy_ret_5d_pct", "spy_ret_20d_pct", "spy_gap_pct"]


def rsi(close: pd.Series, n: int = 14) -> float | None:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean().replace(0, 1e-10)
    v = float((100 - 100 / (1 + gain / loss)).iloc[-1])
    return None if math.isnan(v) else v


def stock_features(t: str, d: date, S: float | None) -> dict:
    df = asof._raw(t)
    if df is None:
        return {}
    hist = df[df.index < pd.Timestamp(d)]
    today = df[df.index == pd.Timestamp(d)]
    if len(hist) < 60:
        return {}
    c = hist["closeadj"].astype(float)
    last = hist.iloc[-1]
    k_prev = float(last["closeunadj"]) / float(last["close"])
    prev_u = float(last["closeunadj"])
    out = {"ret_1d_pct": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
           "ret_5d_pct": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
           "ret_20d_pct": (c.iloc[-1] / c.iloc[-21] - 1) * 100,
           "ret_60d_pct": (c.iloc[-1] / c.iloc[-61] - 1) * 100,
           "dist_ma20_pct": (c.iloc[-1] / c.iloc[-20:].mean() - 1) * 100,
           "dist_ma50_pct": (c.iloc[-1] / c.iloc[-50:].mean() - 1) * 100 if len(c) >= 50 else None,
           "dist_52w_high_pct": (c.iloc[-1] / c.iloc[-252:].max() - 1) * 100,
           "dist_52w_low_pct": (c.iloc[-1] / c.iloc[-252:].min() - 1) * 100,
           "prev_range_pct": (float(last["high"]) - float(last["low"])) / float(last["close"]) * 100,
           "vol_ratio_prev": float(last["volume"]) / float(hist["volume"].iloc[-21:-1].mean()),
           "stock_price": prev_u}
    r = rsi(c)
    out["rsi_minus_50"] = r - 50 if r is not None else None
    hl = (hist["high"] - hist["low"]).astype(float)
    pc = hist["close"].shift(1).astype(float)
    tr = pd.concat([hl, (hist["high"] - pc).abs(), (hist["low"] - pc).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = float(tr.iloc[-14:].mean() / float(last["close"]) * 100)
    if len(today):
        row = today.iloc[0]
        open_u = float(row["open"]) * float(row["closeunadj"]) / float(row["close"])
        out["gap_pct"] = (open_u / prev_u - 1) * 100
        if S:
            out["early_move_pct"] = (S / open_u - 1) * 100
    return out


def earnings_features(t: str, d: date, end_day: date) -> dict:
    e = asof.earnings_dates(t)
    nxt = [x for x in e if x >= d]
    prv = [x for x in e if x < d]
    out = {"days_to_earnings": (nxt[0] - d).days if nxt else None,
           "days_since_earnings": (d - prv[-1]).days if prv else None}
    out["earnings_in_window"] = int(bool(nxt) and nxt[0] <= end_day)
    return out


_SPY: dict[date, dict] = {}


def spy_features(d: date) -> dict:
    if d not in _SPY:
        f = stock_features("SPY", d, None)
        df = asof._raw("SPY")
        hist = df[df.index < pd.Timestamp(d)]["closeadj"].astype(float)
        _SPY[d] = {"spy_vs_ma20_pct": f.get("dist_ma20_pct"), "spy_ret_5d_pct": f.get("ret_5d_pct"),
                   "spy_ret_20d_pct": f.get("ret_20d_pct"), "spy_gap_pct": f.get("gap_pct"),
                   "spy_vol_20d_pct": float(hist.pct_change().iloc[-20:].std() * math.sqrt(252) * 100)}
    return _SPY[d]


def main():
    d = pd.read_csv(R / "calm_test.csv")
    d = d[d["peak_pct"].notna()].copy()
    rows = []
    for _, r in d.iterrows():
        day = date.fromisoformat(r["entry_date"])
        f = {}
        f.update(stock_features(r["ticker"], day, r["stock_at_entry"]))
        f.update(earnings_features(r["ticker"], day, date.fromisoformat(r["end_day"])))
        f.update(spy_features(day))
        sign = 1 if r["side"] == "CALL" else -1
        for k in SIGNED:
            if f.get(k) is not None:
                f[f"{k}_with"] = sign * f[k]
        S, K = r["stock_at_entry"], r["strike"]
        f.update({"iv_over_rv20": r["iv_pct"] / r["vol_20d_pct"] if r["vol_20d_pct"] else None,
                  "premium_dollars": r["ask"] * 100,
                  "premium_pct_of_stock": r["mid"] / S * 100,
                  "otm_pct": ((K / S - 1) if sign > 0 else (1 - K / S)) * 100})
        rows.append(f)
    out = pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out.to_csv(R / "calm_features.csv", index=False)
    print(f"::notice title=Calm features::{len(out)} options, {out.shape[1]} columns -> "
          f"results/analysis/calm_features.csv", flush=True)


if __name__ == "__main__":
    main()
