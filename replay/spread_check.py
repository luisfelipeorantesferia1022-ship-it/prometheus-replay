"""
Check of the winner-search lead (2026-10-22): do options bought with an entry
bid-ask spread under 5% really do better? Same 1,518 calm-test options, now
scored under the actual sell rules (live V3 at 1-minute precision, V3 + day-2
close, day-2 close only, hold to the Thursday exit, +50% take-profit), with a
week-clustered bootstrap and random-subset comparison.

    python -m replay.spread_check     # prints; writes results/analysis/spread_check.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import exit_lab as e

R = e.R
RULES = ["tp50", "v3", "v3_day2", "day2", "hold"]


def build() -> pd.DataFrame:
    P, meta = e.load()
    f = pd.read_csv(R / "calm_features.csv")
    f["key"] = f["raw_symbol"] + "|" + f["entry_date"]
    f = f.set_index("key").reindex(P.keys)
    ex = pd.read_csv(R / "exit_v3_check.csv").set_index("key").reindex(P.keys)
    d = pd.DataFrame(index=pd.Index(P.keys, name="key"))
    d["half"], d["tradable"] = P.half, P.tradable
    for c in ["asof", "ticker", "side", "group", "spread_pct", "delta", "stock_price", "peak_pct", "end_pct"]:
        d[c] = f[c].to_numpy()
    d["tp50"] = np.where(d["peak_pct"] >= 50, 50.0, d["end_pct"])
    d["v3"] = ex["v3_pct"].to_numpy()                      # exact, 1-minute
    d["v3_day2"], _ = e.simulate(P, sl=-75, trail=(50, 20), tday=2)
    d["day2"], _ = e.simulate(P, tday=2)
    d["hold"], _ = e.simulate(P)
    return d


def boot(x: pd.DataFrame, col: str, rng, B: int = 20000):
    wk = x.groupby("asof")[col].agg(["sum", "count"])
    S, C = wk["sum"].to_numpy(), wk["count"].to_numpy()
    idx = rng.integers(0, len(wk), (B, len(wk)))
    m = S[idx].sum(1) / C[idx].sum(1)
    return np.percentile(m, [5, 95]).round(1), round(float((m > 0).mean()), 3)


def main():
    d = build()
    d.to_csv(R / "spread_check.csv")
    t = d[d["tradable"]]
    bands = pd.cut(t["spread_pct"], [0, 2, 3, 4, 5, 6, 7, 8, 10, 12.5, 15, 17.5, 20.01], right=False)
    print(t.groupby(bands, observed=True)[RULES].mean().round(1).assign(n=t.groupby(bands, observed=True).size()))
    s = d[d["spread_pct"] < 5]
    print(f"\n<5%: {len(s)} options, {s['ticker'].nunique()} tickers, {s['asof'].nunique()} weeks")
    print(s.groupby("half")[RULES].mean().round(1).assign(n=s.groupby("half").size()))
    rng = np.random.default_rng(0)
    for r in RULES:
        print(f"  {r:8s} 90% CI (week-clustered) {boot(s, r, rng)}")
    v = s["v3"].sort_values()
    print(f"V3: drop top 3 {v.iloc[:-3].mean():.1f}, drop top 5 {v.iloc[:-5].mean():.1f}")
    crash = (s["asof"] >= "2025-03-07") & (s["asof"] <= "2025-04-25")
    print(f"V3 without Mar 7-Apr 25: {s.loc[~crash, 'v3'].mean():.1f} "
          f"(H1 rest {s.loc[~crash & (s['half'] == 'H1'), 'v3'].mean():.1f})")
    ps = np.array([t["v3"].sample(len(s), random_state=i).mean() for i in range(5000)])
    print(f"random {len(s)} tradable options, V3 mean 99th pct {np.percentile(ps, 99):.1f}")
    o = d[d["ticker"].isin(set(s["ticker"])) & (d["spread_pct"] >= 5) & d["tradable"]]
    print(f"same tickers when spread 5-20%: n={len(o)} V3 {o['v3'].mean():.1f}")


if __name__ == "__main__":
    main()
