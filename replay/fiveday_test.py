"""
Five-day direction test, pre-registered 2026-10-23 (results/fiveday/PREREGISTRATION.md).

The oracle test (replay/oracle_test.py) showed V3 is paid for the stock's
direction over about the next five trading days: knowing it perfectly picks the
winning option side 84% of the time, and a forecaster needs roughly 55% accuracy
to break even and 60% to make about +4% per trade. This asks whether anything
known by Friday's close calls that direction, on every S&P 500 stock-week
(point-in-time membership) from 2013 to September 2026, using Sharadar daily data.

Each week: as-of = last trading day of the ISO week; entry = next session's open
(day 0); exit = the close five sessions later (day 5); return on the dividend-
and split-adjusted basis. Each candidate scores every eligible stock; its 10
strongest scores that week are its calls (direction = sign of the score).

    python -m replay.fiveday_test --data DIR --check    # mechanics only, as-of 2010-2012
    python -m replay.fiveday_test --data DIR            # score 2013-2026 -> results/fiveday/
DIR holds sp500_members_daily_2009_2026.csv.gz, spy_daily.csv.gz and
sp500_membership_raw.csv (derived from the Sharadar bulk tables; not committed).
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

OUT = config.RESULTS_DIR / "fiveday"

# ── Locked by the pre-registration ───────────────────────────────────────────
TEST_FIRST, TEST_LAST = "2013-01-01", "2026-09-25"
CHECK_FIRST, CHECK_LAST = "2010-01-01", "2012-12-31"
TOP_K = 10
MIN_PRICE, MIN_DV20, MIN_HISTORY = 5.0, 20e6, 260
EMBARGO_DAYS = 14
PASS = {"pooled_acc": 0.60, "min_year_acc": 0.55, "ci_low": 0.57, "excess": 0.05}
BOOT, SEED = 20000, 0
GBM = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
           l2_regularization=1.0, random_state=0)
FEATURES = ["r1", "r5", "r20", "r60", "r120", "r250", "ma20", "ma50", "ma200", "vol20", "vol60",
            "vol_ratio", "dv_ratio", "rsi14", "hi52", "beta", "spy_r5", "spy_r20", "spy_ma20",
            "spy_vol20", "rk_r5", "rk_r20", "rk_vol20"]
CANDIDATES = ["K1_trend20", "K2_trend60", "K3_reversal5", "K4_regime_beta", "K5_gbm"]


def load(d: Path):
    df = pd.read_csv(d / "sp500_members_daily_2009_2026.csv.gz", parse_dates=["date"])
    spy = pd.read_csv(d / "spy_daily.csv.gz", parse_dates=["date"]).set_index("date").sort_index()
    sp = pd.read_csv(d / "sp500_membership_raw.csv", parse_dates=["date"])
    cal = spy.index
    w = {k: df.pivot(index="date", columns="ticker", values=k).reindex(cal) for k in
         ("open", "close", "volume", "closeadj", "closeunadj")}
    return w, spy, sp


def features(w, spy) -> dict:
    C = w["closeadj"]
    O = w["open"] * C / w["close"]                       # open on the same adjusted basis
    lr = np.log(C).diff()
    f = {f"r{k}": C / C.shift(k) - 1 for k in (1, 5, 20, 60, 120, 250)}
    for k in (20, 50, 200):
        f[f"ma{k}"] = C / C.rolling(k, min_periods=k).mean() - 1
    f["vol20"] = lr.rolling(20, min_periods=20).std()
    f["vol60"] = lr.rolling(60, min_periods=60).std()
    f["vol_ratio"] = f["vol20"] / f["vol60"]
    dv = w["close"] * w["volume"]
    f["dv20"] = dv.rolling(20, min_periods=15).mean()
    f["dv_ratio"] = dv.rolling(5, min_periods=4).mean() / dv.rolling(60, min_periods=45).mean()
    d = C.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean().replace(0, 1e-10)
    f["rsi14"] = 100 - 100 / (1 + gain / loss)
    f["hi52"] = C / C.rolling(250, min_periods=200).max() - 1
    s = np.log(spy["closeadj"]).diff()
    ers = lr.mul(s, axis=0).rolling(250, min_periods=200).mean()
    er = lr.rolling(250, min_periods=200).mean()
    es = s.rolling(250, min_periods=200).mean()
    vs = s.rolling(250, min_periods=200).var(ddof=0)
    f["beta"] = (ers - er.mul(es, axis=0)).div(vs, axis=0)
    sc = spy["closeadj"]
    f["_spy"] = pd.DataFrame({"spy_r5": sc / sc.shift(5) - 1, "spy_r20": sc / sc.shift(20) - 1,
                              "spy_ma20": sc / sc.rolling(20).mean() - 1,
                              "spy_vol20": s.rolling(20).std()})
    f["_hist"] = C.notna().cumsum()
    f["_C"], f["_O"], f["_P"] = C, O, w["closeunadj"]
    return f


def stock_weeks(f, sp, first: str, last: str) -> pd.DataFrame:
    cal = f["_C"].index
    iso = pd.Series(cal, index=cal).groupby([cal.isocalendar().year.values, cal.isocalendar().week.values]).max()
    asofs = [pd.Timestamp(a) for a in sorted(iso.values) if pd.Timestamp(first) <= pd.Timestamp(a) <= pd.Timestamp(last)]
    pos = {d: i for i, d in enumerate(cal)}
    C, O = f["_C"], f["_O"]
    rows = []
    for a in asofs:
        i = pos[a]
        if i + 6 >= len(cal):
            continue
        mem = [t for t in sp500_members(sp, a.date()) if t in C.columns]
        e0, e5 = cal[i + 1], cal[i + 6]
        x = pd.DataFrame(index=pd.Index(mem, name="ticker"))
        x["price"] = f["_P"].loc[a, mem]
        x["dv20"] = f["dv20"].loc[a, mem]
        x["hist"] = f["_hist"].loc[a, mem]
        x["open0"] = O.loc[e0, mem]
        x["close5"] = C.loc[e0:e5, mem].ffill().iloc[-1]          # delisted mid-week: last close
        x = x[(x["price"] >= MIN_PRICE) & (x["dv20"] >= MIN_DV20) & (x["hist"] >= MIN_HISTORY)
              & x["open0"].notna() & x["close5"].notna()]
        if x.empty:
            continue
        for k in FEATURES:
            if k.startswith("spy_"):
                x[k] = f["_spy"].loc[a, k]
            elif not k.startswith("rk_"):
                x[k] = f[k].loc[a, x.index]
        for k in ("r5", "r20", "vol20"):
            x[f"rk_{k}"] = x[k].rank(pct=True)
        x["ret"] = x["close5"] / x["open0"] - 1
        x["asof"], x["entry"], x["exit"] = a, e0, e5
        rows.append(x.reset_index())
    sw = pd.concat(rows, ignore_index=True)
    sw["up"] = (sw["ret"] > 0).astype(int)
    sw["base_up"] = sw.groupby("asof")["up"].transform("mean")
    return sw


def gbm_scores(sw: pd.DataFrame, test_years: list[int]) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = pd.Series(np.nan, index=sw.index)
    for y in test_years:
        cut = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)
        tr = sw[(sw["asof"] < cut) & sw["r250"].notna()]
        te = sw["asof"].dt.year == y
        m = HistGradientBoostingClassifier(**GBM).fit(tr[FEATURES], tr["up"])
        out[te] = m.predict_proba(sw.loc[te, FEATURES])[:, 1] - 0.5
        print(f"[fiveday] gbm {y}: trained on {len(tr):,} stock-weeks to {tr['asof'].max().date()}", flush=True)
    return out


def scores(sw: pd.DataFrame, test_years: list[int]) -> pd.DataFrame:
    s = pd.DataFrame(index=sw.index)
    s["K1_trend20"] = sw["r20"]
    s["K2_trend60"] = sw["r60"]
    s["K3_reversal5"] = -sw["r5"]
    s["K4_regime_beta"] = np.where(sw["spy_ma20"] >= 0, 1.0, -1.0) * sw["beta"]
    s["K5_gbm"] = gbm_scores(sw, test_years)
    return s


def calls(sw: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in CANDIDATES:
        x = sw[["asof", "ticker", "ret", "up", "base_up"]].assign(score=s[c]).dropna(subset=["score"])
        x = x[x["score"] != 0]
        x = x.assign(conf=x["score"].abs()).sort_values(["asof", "conf", "ticker"], ascending=[True, False, True])
        x = x.groupby("asof").head(TOP_K).copy()
        x["dir"] = np.sign(x["score"])
        x["correct"] = np.where(x["dir"] > 0, x["ret"] > 0, x["ret"] < 0).astype(int)
        x["base"] = np.where(x["dir"] > 0, x["base_up"], 1 - x["base_up"])
        out.append(x.assign(candidate=c))
    return pd.concat(out, ignore_index=True)


def boot_ci(x: pd.DataFrame) -> list[float]:
    wk = x.groupby("asof")["correct"].agg(["sum", "count"])
    S, N = wk["sum"].to_numpy(), wk["count"].to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(wk), (BOOT, len(wk)))
    m = S[idx].sum(1) / N[idx].sum(1)
    return [round(float(v), 4) for v in np.percentile(m, [5, 95])]


def score_candidate(x: pd.DataFrame) -> dict:
    yr = x.groupby(x["asof"].dt.year)["correct"].mean()
    acc, excess = float(x["correct"].mean()), float((x["correct"] - x["base"]).mean())
    ci = boot_ci(x)
    lng, sht = x[x["dir"] > 0], x[x["dir"] < 0]
    c = {"P1_pooled_acc": acc >= PASS["pooled_acc"], "P2_every_year": bool((yr >= PASS["min_year_acc"]).all()),
         "P3_ci_low": ci[0] >= PASS["ci_low"], "P4_excess": excess >= PASS["excess"]}
    return {"calls": int(len(x)), "weeks": int(x["asof"].nunique()), "accuracy": round(acc, 4), "ci90": ci,
            "excess_over_base": round(excess, 4), "worst_year": [int(yr.idxmin()), round(float(yr.min()), 4)],
            "by_year": {int(k): round(float(v), 4) for k, v in yr.items()},
            "long_share": round(len(lng) / len(x), 3),
            "long_acc": round(float(lng["correct"].mean()), 4) if len(lng) else None,
            "short_acc": round(float(sht["correct"].mean()), 4) if len(sht) else None,
            "criteria": c, "verdict": "pass" if all(c.values()) else "fail"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--detail", default="fiveday_calls.parquet", help="per-call file (stock-level, kept off the repo)")
    a = ap.parse_args()
    w, spy, sp = load(Path(a.data))
    f = features(w, spy)
    if a.check:
        sw = stock_weeks(f, sp, CHECK_FIRST, CHECK_LAST)
        print(f"[check] {sw['asof'].nunique()} weeks, {len(sw):,} stock-weeks, {sw['ticker'].nunique()} tickers, "
              f"per week {sw.groupby('asof').size().describe()[['min', 'mean', 'max']].round(0).to_dict()}")
        print("[check] feature NaN share:", sw[FEATURES].isna().mean().round(3).to_dict())
        print(f"[check] ret: mean {sw['ret'].mean() * 100:.2f}%, |ret|>50%: {(sw['ret'].abs() > .5).sum()}")
        return
    sw = stock_weeks(f, sp, "2010-01-01", TEST_LAST)
    years = list(range(int(TEST_FIRST[:4]), int(TEST_LAST[:4]) + 1))
    s = scores(sw, years)
    test = sw["asof"] >= pd.Timestamp(TEST_FIRST)
    cl = calls(sw[test], s[test])
    res = {"locked": {"top_k": TOP_K, "pass": PASS, "test": [TEST_FIRST, TEST_LAST], "candidates": CANDIDATES,
                      "gbm": GBM, "features": FEATURES},
           "universe": {"weeks": int(sw.loc[test, "asof"].nunique()), "stock_weeks": int(test.sum()),
                        "share_up": round(float(sw.loc[test, "up"].mean()), 4),
                        "share_up_by_year": sw[test].groupby(sw.loc[test, "asof"].dt.year)["up"].mean().round(4)
                                            .rename(int).to_dict()},
           "candidates": {c: score_candidate(cl[cl["candidate"] == c]) for c in CANDIDATES}}
    v = [x["verdict"] for x in res["candidates"].values()]
    res["verdict"] = "a candidate passed" if "pass" in v else "no candidate passed"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2, default=str))
    cl.to_parquet(a.detail, index=False)          # stock-level detail stays off the repo
    for c, r in res["candidates"].items():
        print(f"{c:16s} acc {r['accuracy']:.3f} CI {r['ci90']} excess {r['excess_over_base']:+.3f} "
              f"worst {r['worst_year']} long {r['long_share']:.2f} -> {r['verdict']} {r['criteria']}")
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
