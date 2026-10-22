"""
Sell-rule lab: replays many sell strategies over the real price paths of the
calm-test options (results/analysis/exit_paths.parquet) and measures what each
would have returned, separately for January-June and July-December so a rule
only counts if it works in both.

Every trade is bought at the ask at 9:46 (as the agent does). Rules:
  take-profit  a resting limit sell at +X%: fills when the bid reaches it
  stop-loss    sell at the bid when the mid is down Y% (V3 style)
  trailing     once the mid has been up T%, sell at the bid when it falls G
               points below its best (V3: T=50, G=20)
  time stop    sell at the bid at the close of trading day N after entry
  always       force-exit 13:00 the day before expiry (V3), as every rule does
The price paths are 5-minute buckets; the live V3 rules are also simulated at
1-minute precision upstream (exit_v3_check.csv) to measure the bucket error.

    python -m replay.exit_lab      # results/analysis/exit_rules.csv, exit_lab.json
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import asof, config

R = config.RESULTS_DIR / "analysis"
SPLIT = "2025-07-01"
TPS = [None, 20, 30, 40, 50, 75, 100, 150, 200]
SLS = [None, -30, -40, -50, -60, -75]
TRAILS = [None, (25, 15), (50, 20), (50, 30), (75, 25), (100, 30), (100, 50)]
TIMES = [None, 0, 1, 2, 3, 5, 7]


@dataclass
class Paths:
    keys: list
    cost: np.ndarray
    half: np.ndarray
    tradable: np.ndarray
    valid: np.ndarray
    bid_h: np.ndarray
    bid_c: np.ndarray
    mid_h: np.ndarray
    mid_c: np.ndarray
    day: np.ndarray        # trading days since entry
    last_of_day: np.ndarray
    force: np.ndarray
    tt: np.ndarray         # trading time since entry, in trading days


def load() -> tuple[Paths, pd.DataFrame]:
    meta = pd.read_csv(R / "calm_test.csv")
    meta = meta[meta["peak_pct"].notna()].copy()
    meta["key"] = meta["raw_symbol"] + "|" + meta["entry_date"]
    p = pd.read_parquet(R / "exit_paths.parquet")
    p = p[p["key"].isin(set(meta["key"]))].sort_values(["key", "ts_et"])
    meta = meta[meta["key"].isin(set(p["key"]))].set_index("key")
    keys = sorted(meta.index)
    days = asof.trading_days(pd.Timestamp("2025-01-01").date(), pd.Timestamp("2026-02-27").date())
    dix = {d: i for i, d in enumerate(days)}
    groups = {k: g for k, g in p.groupby("key")}
    L = max(len(g) for g in groups.values())
    n = len(keys)
    arr = {c: np.full((n, L), np.nan, dtype="float64") for c in ["bid_h", "bid_c", "mid_h", "mid_c", "tt"]}
    day = np.full((n, L), -1, dtype="int32")
    valid = np.zeros((n, L), bool)
    last_of_day = np.zeros((n, L), bool)
    force = np.zeros((n, L), bool)
    for i, k in enumerate(keys):
        g = groups[k]
        m = len(g)
        ts = pd.to_datetime(g["ts_et"])
        ed = pd.Timestamp(meta.at[k, "entry_date"]).date()
        di = np.array([dix[x] for x in ts.dt.date]) - dix[ed]
        for c in ["bid_h", "bid_c", "mid_h", "mid_c"]:
            arr[c][i, :m] = g[c].to_numpy()
        mins = (ts.dt.hour * 60 + ts.dt.minute - 570).to_numpy()
        arr["tt"][i, :m] = di + mins / 390
        day[i, :m] = di
        valid[i, :m] = True
        ld = np.r_[di[1:] != di[:-1], True]
        last_of_day[i, :m] = ld
        fday = pd.Timestamp(meta.at[k, "expiry"]).date() - pd.Timedelta(days=1)
        force[i, :m] = (ts.dt.date >= fday).to_numpy() & (ts.dt.hour >= config.EXIT_FORCE_HOUR_ET).to_numpy()
    P = Paths(keys=keys, cost=meta.loc[keys, "ask"].to_numpy(float),
              half=np.where(meta.loc[keys, "asof"].to_numpy() < SPLIT, "H1", "H2"),
              tradable=((meta.loc[keys, "spread_pct"] <= 20) & (meta.loc[keys, "delta"].abs() >= 0.35)).to_numpy(),
              valid=valid, bid_h=arr["bid_h"], bid_c=arr["bid_c"], mid_h=arr["mid_h"], mid_c=arr["mid_c"],
              day=day, last_of_day=last_of_day, force=force, tt=arr["tt"])
    return P, meta.loc[keys]


def simulate(P: Paths, tp=None, sl=None, trail=None, tday=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (% return per trade, exit bucket index)."""
    n, L = P.bid_c.shape
    cost = P.cost
    done = np.zeros(n, bool)
    ret = np.full(n, np.nan)
    at = np.full(n, -1)
    peak = np.full(n, -np.inf)
    tp_px = cost * (1 + tp / 100) if tp is not None else None
    last_valid = P.valid.sum(axis=1) - 1
    for j in range(L):
        v = P.valid[:, j] & ~done
        if not v.any():
            break
        if tp_px is not None:
            hit = v & (P.bid_h[:, j] >= tp_px)
            ret[hit], at[hit] = tp, j
            done |= hit
            v &= ~hit
        pk = (P.mid_h[:, j] / cost - 1) * 100
        peak = np.where(v, np.fmax(peak, pk), peak)
        cur = (P.mid_c[:, j] / cost - 1) * 100
        out = np.zeros(n, bool)
        if sl is not None:
            out |= cur <= sl
        if trail is not None:
            out |= (peak >= trail[0]) & (cur <= peak - trail[1])
        if tday is not None:
            out |= (P.day[:, j] >= tday) & P.last_of_day[:, j]
        out |= P.force[:, j] | (j == last_valid)
        out &= v
        ret[out] = (P.bid_c[out, j] / cost[out] - 1) * 100
        at[out] = j
        done |= out
    return ret, at


def stats(r: np.ndarray) -> dict:
    r = r[~np.isnan(r)]
    if not len(r):
        return {}
    w = r[r > 0]
    lo = r[r <= 0]
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)),
            "win_rate": float((r > 0).mean() * 100), "avg_win": float(w.mean()) if len(w) else 0.0,
            "avg_loss": float(lo.mean()) if len(lo) else 0.0,
            "profit_factor": float(w.sum() / -lo.sum()) if lo.sum() < 0 else float("inf")}


def label(tp, sl, trail, tday) -> str:
    parts = []
    if tp is not None:
        parts.append(f"take profit +{tp}%")
    if sl is not None:
        parts.append(f"stop {sl}%")
    if trail is not None:
        parts.append(f"trail {trail[1]} pts after +{trail[0]}%")
    if tday is not None:
        parts.append("sell at entry-day close" if tday == 0 else f"sell at day-{tday} close")
    return ", ".join(parts) or "hold to the Thursday exit"


def grid(P: Paths) -> pd.DataFrame:
    rows = []
    for tp, sl, trail, tday in itertools.product(TPS, SLS, TRAILS, TIMES):
        ret, _ = simulate(P, tp, sl, trail, tday)
        row = {"rule": label(tp, sl, trail, tday), "tp": tp, "sl": sl,
               "trail": f"{trail[0]}/{trail[1]}" if trail else None, "time_day": tday}
        for name, m in (("all", np.ones(len(ret), bool)), ("tradable", P.tradable)):
            for h in ("H1", "H2"):
                s = stats(ret[m & (P.half == h)])
                for k, v in s.items():
                    row[f"{name}_{h}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    for name in ("all", "tradable"):
        df[f"{name}_worst_half_mean"] = df[[f"{name}_H1_mean", f"{name}_H2_mean"]].min(axis=1)
    return df


def patterns(P: Paths, meta: pd.DataFrame) -> dict:
    out = {}
    m = P.tradable
    # median / quartile path of the bid P&L over trading time (5-minute grid, 0..9 trading days)
    grid_t = np.round(np.arange(16 / 390, 9.0, 5 / 390), 4)
    pnl = (P.bid_c / P.cost[:, None] - 1) * 100
    tt = np.round(P.tt, 4)

    def band(sel):
        res = []
        for t in grid_t:
            vals = pnl[sel][np.isclose(tt[sel], t, atol=1e-3)]
            if len(vals) >= 30:
                res.append([float(t), *np.percentile(vals, [25, 50, 75]).round(1).tolist(), int(len(vals))])
        return res
    win = (meta["peak_pct"].to_numpy() >= 50)
    out["path_all"] = band(m)
    out["path_win50"] = band(m & win)
    out["path_never50"] = band(m & ~win)
    # when peaks happen
    d = meta[P.tradable]
    pk = pd.to_datetime(d["peak_at"])
    out["peak_day"] = d["days_to_peak"].clip(upper=10).value_counts().sort_index().to_dict()
    out["peak_halfhour"] = ((pk.dt.hour * 60 + pk.dt.minute - 570) // 30).value_counts().sort_index().to_dict()
    # what follows the first touch of +X
    after = {}
    for x in (25, 50, 100):
        rows = []
        for i in np.where(m)[0]:
            v = P.valid[i]
            bh = P.bid_h[i, v]
            hit = np.where(bh >= P.cost[i] * (1 + x / 100))[0]
            if not len(hit):
                continue
            j = hit[0]
            later_max = (bh[j:].max() / P.cost[i] - 1) * 100
            end = (P.bid_c[i, v][-1] / P.cost[i] - 1) * 100
            rows.append((later_max, end, P.tt[i, v][j]))
        a = np.array(rows) if rows else np.zeros((0, 3))
        after[x] = {"reached": int(len(a)), "share_of_trades": float(len(a) / m.sum() * 100),
                    "went_on_to_double_target": float((a[:, 0] >= 2 * x).mean() * 100) if len(a) else None,
                    "ended_below_zero": float((a[:, 1] < 0).mean() * 100) if len(a) else None,
                    "ended_below_target": float((a[:, 1] < x).mean() * 100) if len(a) else None,
                    "median_end": float(np.median(a[:, 1])) if len(a) else None,
                    "median_days_to_reach": float(np.median(a[:, 2])) if len(a) else None}
    out["after_touch"] = after
    return out


def main():
    P, meta = load()
    print(f"[lab] {len(P.keys)} options, {P.tradable.sum()} agent-tradable, path length {P.bid_c.shape[1]}")
    g = grid(P)
    g.to_csv(R / "exit_rules.csv", index=False)
    v3 = g[(g["tp"].isna()) & (g["sl"] == -75) & (g["trail"] == "50/20") & (g["time_day"].isna())]
    exact = pd.read_csv(R / "exit_v3_check.csv")
    exact = exact.set_index("key").reindex(P.keys)
    out = {"n": len(P.keys), "tradable": int(P.tradable.sum()),
           "v3_5min": v3.iloc[0].to_dict() if len(v3) else None,
           "v3_exact_1min": {f"{nm}_{h}": stats(exact["v3_pct"].to_numpy()[mk & (P.half == h)])
                             for nm, mk in (("all", np.ones(len(P.keys), bool)), ("tradable", P.tradable))
                             for h in ("H1", "H2")},
           "patterns": patterns(P, meta)}
    (R / "exit_lab.json").write_text(json.dumps(out, indent=1, default=lambda o: None if o != o else str(o)))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["rule", "tradable_H1_mean", "tradable_H2_mean", "tradable_H1_win_rate", "tradable_H2_win_rate",
            "all_H1_mean", "all_H2_mean"]
    print(g.sort_values("tradable_worst_half_mean", ascending=False)[cols].head(25).round(1).to_string())
    print("\nV3 (5-min sim):", v3[cols].round(1).to_string())
    print("V3 exact:", json.dumps(out["v3_exact_1min"], indent=0)[:600])


if __name__ == "__main__":
    main()
