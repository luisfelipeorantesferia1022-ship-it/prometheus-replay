"""
Twin-stock pairs test, pre-registered 2026-10-23 (results/pairs/PREREGISTRATION.md).

Two variants, same trading rule, stocks only (no options), daily Sharadar data:
  P1 twins     20 economically linked pairs named in advance (Luis's six:
               KO/PEP, HD/LOW, V/MA, WMT/TGT, NKE/UAA, UPS/FDX, plus 14 same-
               industry twins)
  P2 emerging  every January and July, pairs among the 150 most-traded S&P 500
               members whose daily returns became tightly linked over the past
               year (corr >= 0.70, up >= 0.20 on the three years before, or too
               new to have that history); the top 20 are traded for six months

Rule: spread = ln(A) - ln(B) on adjusted closes; z = (spread - 60-day mean) /
60-day sd. When |z| >= 2 at a close, at the next open buy the cheap one and
short the rich one with equal dollars. Exit at the next open after z crosses 0,
|z| reaches 4 (stop), 20 sessions pass, either stock leaves the S&P 500, or the
P2 period ends. After any exit a pair needs |z| < 1 before it can trade again.
Return per trade on gross notional: (long leg - short leg) / 2, net of 0.15%
round-trip cost and 0.5%/yr borrow on the short half.

    python -m replay.pairs_test --data DIR --check   # mechanics, 2010-2012, no P&L shown
    python -m replay.pairs_test --data DIR           # score 2013 .. data end -> results/pairs/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .asof import sp500_members
from .fiveday_test import load

OUT = config.RESULTS_DIR / "pairs"

# ── Locked by the pre-registration ───────────────────────────────────────────
TEST_FIRST = "2013-01-01"
SPLIT = "2020-01-01"
TWINS = [("KO", "PEP"), ("HD", "LOW"), ("V", "MA"), ("WMT", "TGT"), ("NKE", "UAA"), ("UPS", "FDX"),
         ("XOM", "CVX"), ("JPM", "BAC"), ("GS", "MS"), ("MCD", "YUM"), ("LMT", "NOC"), ("CSX", "NSC"),
         ("DUK", "SO"), ("PG", "CL"), ("MO", "PM"), ("T", "VZ"), ("ADP", "PAYX"), ("SPGI", "MCO"),
         ("HLT", "MAR"), ("DAL", "UAL")]
LUIS_PAIRS = TWINS[:6]
LOOKBACK, ENTRY_Z, STOP_Z, RESET_Z, MAX_HOLD = 60, 2.0, 4.0, 1.0, 20
COST_RT, BORROW_YR = 0.0015, 0.005
FORM_MONTHS, TOP_LIQ, CORR_NEW_MIN, CORR_GAIN_MIN = (1, 7), 150, 0.70, 0.20
NEW_WIN, OLD_WIN, OLD_MIN_OBS, NEW_MIN_OBS, MAX_PAIRS = 252, 756, 500, 200, 20
PASS = {"min_trades": 100, "t_clustered": 2.0, "drop_top_share": 0.03, "max_pair_share": 0.25}


# ─────────────────────────────────────────────────────────────────────────────

def membership_by_day(cal: pd.DatetimeIndex, sp: pd.DataFrame) -> dict:
    """Members known as of the previous week's last session (point in time)."""
    iso = pd.Series(cal, index=cal).groupby([cal.isocalendar().year.values, cal.isocalendar().week.values]).max()
    fridays = sorted(pd.Timestamp(x) for x in iso.values)
    mem = {f: sp500_members(sp, f.date()) for f in fridays}
    out, cur, j = {}, set(), 0
    for d in cal:
        while j < len(fridays) and fridays[j] < d:
            cur = mem[fridays[j]]; j += 1
        out[d] = cur
    return out


def trade_pair(a: str, b: str, C: pd.DataFrame, O: pd.DataFrame, mem: dict, start, end, tag: str) -> list[dict]:
    """Simulate the rule on one pair between start and end (inclusive session dates)."""
    s = np.log(C[a]) - np.log(C[b])
    mu, sd = s.rolling(LOOKBACK, min_periods=LOOKBACK).mean(), s.rolling(LOOKBACK, min_periods=LOOKBACK).std()
    z = ((s - mu) / sd).to_numpy()
    cal = C.index
    i0, i1 = cal.searchsorted(pd.Timestamp(start)), cal.searchsorted(pd.Timestamp(end), side="right") - 1
    oa, ob, ca, cb = O[a].to_numpy(), O[b].to_numpy(), C[a].to_numpy(), C[b].to_numpy()
    gate = np.array([a in mem[d] and b in mem[d] for d in cal])
    trades, pos, armed = [], None, True
    for i in range(i0, i1 + 1):
        zi = z[i]
        if pos is None:
            if not np.isfinite(zi):
                continue
            if abs(zi) < RESET_Z:
                armed = True
            if (armed and abs(zi) >= ENTRY_Z and gate[i] and i + 1 <= i1 and gate[i + 1]
                    and np.isfinite(oa[i + 1]) and np.isfinite(ob[i + 1])):
                pos = {"e": i + 1, "sgn": 1.0 if zi > 0 else -1.0, "z_entry": float(zi)}
                armed = False
            continue
        if i < pos["e"]:
            continue
        held = i - pos["e"] + 1
        reason = None
        if np.isfinite(zi) and pos["sgn"] * zi <= 0:
            reason = "converged"
        elif np.isfinite(zi) and pos["sgn"] * zi >= STOP_Z:
            reason = "stop"
        elif held >= MAX_HOLD:
            reason = "time"
        elif i + 1 > i1:
            reason = "period end"
        elif not gate[i + 1]:
            reason = "left S&P 500"
        if reason is None:
            continue
        x = i + 1 if i + 1 < len(cal) and np.isfinite(oa[i + 1]) and np.isfinite(ob[i + 1]) else None
        pa = oa[x] if x is not None else ca[i]
        pb = ob[x] if x is not None else cb[i]
        e = pos["e"]
        ra, rb = pa / oa[e] - 1, pb / ob[e] - 1
        # sgn > 0: A rich -> short A, long B
        gross = (rb - ra) / 2 if pos["sgn"] > 0 else (ra - rb) / 2
        sessions = (x if x is not None else i) - e
        net = gross - COST_RT - BORROW_YR * 0.5 * sessions / 252
        trades.append({"variant": tag, "pair": f"{a}/{b}", "entry": cal[e], "exit": cal[x] if x is not None else cal[i],
                       "sessions": int(sessions), "z_entry": round(pos["z_entry"], 3), "reason": reason,
                       "long": b if pos["sgn"] > 0 else a, "gross": gross, "net": net})
        pos, armed = None, False
    return trades


def emerging_pairs(C: pd.DataFrame, dv: pd.DataFrame, mem: dict, names: dict, f: pd.Timestamp) -> list[tuple]:
    cal = C.index
    i = cal.searchsorted(f)
    m = [t for t in mem[cal[i]] if t in C.columns]
    liq = dv.iloc[max(0, i - 60):i][m].mean().dropna().nlargest(TOP_LIQ).index
    r = np.log(C[liq].iloc[max(0, i - NEW_WIN - OLD_WIN):i]).diff()
    new, old = r.iloc[-NEW_WIN:], r.iloc[:-NEW_WIN]
    cn = new.corr(min_periods=NEW_MIN_OBS)
    co = old.corr(min_periods=OLD_MIN_OBS) if len(old) else pd.DataFrame(np.nan, index=liq, columns=liq)
    out = []
    t = list(liq)
    for x in range(len(t)):
        for y in range(x + 1, len(t)):
            a, b = t[x], t[y]
            c1 = cn.at[a, b]
            if not np.isfinite(c1) or c1 < CORR_NEW_MIN:
                continue
            if names.get(a) and names.get(a) == names.get(b):          # share classes of one company
                continue
            c0 = co.at[a, b] if a in co.index and b in co.columns else np.nan
            if np.isfinite(c0) and c0 > c1 - CORR_GAIN_MIN:
                continue
            out.append((a, b, float(c1), float(c0) if np.isfinite(c0) else None))
    return sorted(out, key=lambda q: -q[2])[:MAX_PAIRS]


def formations(cal: pd.DatetimeIndex, first: str, last) -> list[pd.Timestamp]:
    out = []
    for y in range(int(first[:4]), pd.Timestamp(last).year + 1):
        for mth in FORM_MONTHS:
            k = cal.searchsorted(pd.Timestamp(f"{y}-{mth:02d}-01"))
            if k < len(cal) and pd.Timestamp(first) <= cal[k] <= pd.Timestamp(last):
                out.append(cal[k])
    return out


def run(C, O, dv, mem, names, first: str, last) -> tuple[pd.DataFrame, list]:
    trades = []
    for a, b in TWINS:
        trades += trade_pair(a, b, C, O, mem, first, last, "P1_twins")
    forms, chosen = formations(C.index, first, last), []
    for k, f in enumerate(forms):
        end = (forms[k + 1] if k + 1 < len(forms) else pd.Timestamp(last))
        end = C.index[C.index.searchsorted(end) - 1] if k + 1 < len(forms) else end
        ps = emerging_pairs(C, dv, mem, names, f)
        chosen.append({"formation": str(f.date()), "pairs": [f"{a}/{b}" for a, b, *_ in ps],
                       "corr_new": [round(c, 3) for _, _, c, _ in ps]})
        for a, b, *_ in ps:
            trades += trade_pair(a, b, C, O, mem, f, end, "P2_emerging")
    return pd.DataFrame(trades), chosen


def clustered_t(x: pd.Series, cluster: pd.Series) -> float:
    e = x - x.mean()
    se = math.sqrt((e.groupby(cluster).sum() ** 2).sum()) / len(x)
    return float(x.mean() / se) if se > 0 else float("nan")


def score_variant(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return {"trades": 0, "verdict": "fail", "criteria": {}}
    n = t["net"]
    wk = t["entry"].dt.strftime("%G-%V")
    tval = clustered_t(n, wk)
    h1, h2 = t[t["entry"] < SPLIT], t[t["entry"] >= SPLIT]
    k = max(1, math.ceil(PASS["drop_top_share"] * len(t)))
    drop = float(n.sort_values().iloc[:-k].mean()) if len(t) > k else float("nan")
    per = t.groupby("pair")["net"].sum()
    share = float(per.abs().max() / per.abs().sum()) if per.abs().sum() > 0 else 1.0
    c = {"1_trades_ge_100": len(t) >= PASS["min_trades"],
         "2_mean_pos_and_t_ge_2": bool(n.mean() > 0 and tval >= PASS["t_clustered"]),
         "3_both_halves_pos": bool(len(h1) and len(h2) and h1["net"].mean() > 0 and h2["net"].mean() > 0),
         "4_pos_without_top_3pct": bool(drop > 0),
         "5_no_pair_over_25pct": share <= PASS["max_pair_share"]}
    return {"trades": int(len(t)), "pairs_traded": int(t["pair"].nunique()),
            "mean_net_pct": round(float(n.mean()) * 100, 3), "mean_gross_pct": round(float(t["gross"].mean()) * 100, 3),
            "median_net_pct": round(float(n.median()) * 100, 3), "win_rate": round(float((n > 0).mean()) * 100, 1),
            "t_clustered": round(tval, 2), "avg_sessions": round(float(t["sessions"].mean()), 1),
            "h1": {"trades": int(len(h1)), "mean_net_pct": round(float(h1["net"].mean()) * 100, 3) if len(h1) else None},
            "h2": {"trades": int(len(h2)), "mean_net_pct": round(float(h2["net"].mean()) * 100, 3) if len(h2) else None},
            "drop_top": {"k": k, "mean_net_pct": round(drop * 100, 3)},
            "max_pair_share": round(share, 3), "top_pair": per.abs().idxmax() if len(per) else None,
            "exits": t["reason"].value_counts().to_dict(),
            "by_year_mean_net_pct": (t.groupby(t["entry"].dt.year)["net"].mean() * 100).round(3).rename(int).to_dict(),
            "criteria": c, "verdict": "pass" if all(c.values()) else "fail"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--detail", default="pairs_trades.csv", help="per-trade file (kept off the repo)")
    a = ap.parse_args()
    w, spy, sp = load(Path(a.data))
    C = w["closeadj"].loc["2009-01-01":]                     # the stock file starts 2009; SPY's calendar earlier
    O = (w["open"] * w["closeadj"] / w["close"]).loc["2009-01-01":]
    dv = (w["close"] * w["volume"]).loc["2009-01-01":]
    names = sp.sort_values("date").groupby("ticker")["name"].last().to_dict()
    mem = membership_by_day(C.index, sp)
    if a.check:
        t, chosen = run(C, O, dv, mem, names, "2010-07-01", "2012-12-31")
        print(f"[check] trades by variant: {t.groupby('variant').size().to_dict()}")
        print(f"[check] exits: {t.groupby('variant')['reason'].value_counts().to_dict()}")
        print(f"[check] sessions held: {t.groupby('variant')['sessions'].describe()[['mean', 'max']].round(1).to_dict()}")
        print(f"[check] P1 trades per pair: {t[t.variant == 'P1_twins'].groupby('pair').size().to_dict()}")
        for c in chosen:
            print(f"[check] formation {c['formation']}: {len(c['pairs'])} pairs, e.g. {c['pairs'][:6]}")
        return
    last = C.index[-1]
    t, chosen = run(C, O, dv, mem, names, TEST_FIRST, last)
    res = {"locked": {"twins": [f"{x}/{y}" for x, y in TWINS], "lookback": LOOKBACK, "entry_z": ENTRY_Z,
                      "stop_z": STOP_Z, "max_hold": MAX_HOLD, "cost_rt": COST_RT, "borrow_yr": BORROW_YR,
                      "pass": PASS, "test": [TEST_FIRST, str(last.date())]},
           "variants": {v: score_variant(t[t["variant"] == v]) for v in ("P1_twins", "P2_emerging")},
           "formations": chosen}
    p1 = t[t["variant"] == "P1_twins"]
    res["P1_by_pair"] = {p: {"trades": int(len(g)), "mean_net_pct": round(float(g["net"].mean()) * 100, 3),
                             "win_rate": round(float((g["net"] > 0).mean()) * 100, 1),
                             "luis": p in [f"{x}/{y}" for x, y in LUIS_PAIRS]}
                         for p, g in p1.groupby("pair")}
    v = [x["verdict"] for x in res["variants"].values()]
    res["verdict"] = "a variant passed" if "pass" in v else "neither variant passed"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2, default=str))
    t.to_csv(a.detail, index=False)
    for k, r in res["variants"].items():
        print(f"{k}: trades {r['trades']} mean net {r.get('mean_net_pct')}% gross {r.get('mean_gross_pct')}% "
              f"t {r.get('t_clustered')} H1 {r.get('h1')} H2 {r.get('h2')} drop-top {r.get('drop_top')} "
              f"pair share {r.get('max_pair_share')} -> {r['verdict']} {r['criteria']}")
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
