"""
Out-of-sample test of the tight-spread lead, pre-registered on 2026-10-22
(results/oos/PREREGISTRATION.md). The 2025 spread check found that options
bought with an entry spread under 5% made money under the live V3 sell rules,
but that line was drawn on the same 2025 data. This runs the identical rule,
unchanged, on two periods it has never seen: 2024 and January-September 2026.

The rule (contract picking from replay/calm_test.py, exits from
replay/exit_paths.v3_rule, both unchanged):
  * first trading day after each weekly as-of Friday, 9:46 ET
  * for each sampled stock, the call AND the put the agent's rule picks
    (12-24 DTE, 14-21 preferred; strike nearest 1% out of the money)
  * qualifies only if (ask - bid) / mid < 5% and Black-Scholes delta >= 0.35
    on the last live quote at or before 9:46
  * bought at the ask; live V3 sell rules every minute (trigger on the mid,
    sell at the bid): stop -75%, ceiling +200%, trail 20 points once up 50%,
    force exit 13:00 ET the day before expiry

Sample per week (point in time, Sharadar): the TOP_N S&P 500 members by
average daily dollar volume over the 20 trading days to the as-of date, plus
RANDOM_N other members drawn at random (seeded by the date).

    python -m replay.oos_spread --universe             # results/oos/universe.csv (Sharadar only)
    python -m replay.oos_spread --shard i --nshards n  # option data for one shard of weeks
    python -m replay.oos_spread --merge                # results/oos/oos_trades.csv + ledger
    python -m replay.oos_spread --score                # verdict vs the locked criteria
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from . import asof, config, dbn
from .agent import bs_delta, implied_vol
from .calm_test import defs_for_day, pick, stock_at
from .exit_paths import v3_rule

OUT = config.RESULTS_DIR / "oos"
PARTS = OUT / "parts"
UNIVERSE = OUT / "universe.csv"
TRADES = OUT / "oos_trades.csv"
VERDICT = OUT / "verdict.json"

# ── Locked by the pre-registration ───────────────────────────────────────────
PERIODS = {             # first as-of, last as-of, second half starts at as-of >=
    "2024": ("2024-01-05", "2024-12-27", "2024-07-01"),
    "2026": ("2026-01-02", "2026-09-25", "2026-05-15"),
}
TOP_N = int(os.environ.get("OOS_TOP_N", "40"))
RANDOM_N = 10
DV_DAYS = 20
SPREAD_MAX_PCT = 5.0
MIN_DELTA = 0.35
TRADABLE_SPREAD_PCT = 20.0        # live V3 cap; 5-20% options are the comparison group
MIN_TRADES = 100                  # fewer qualifying trades in a period -> no verdict there
DROP_TOP_SHARE = 0.03
BOOT = 20000
BOOT_SEED = 0

ENTRY = time(9, 46)
ONE_MIN = pd.Timedelta(minutes=1)
QUOTE_CHUNK = 8
DEF_CHUNK = 10


# ─────────────────────────────────────────────────────────────────────────────
#  Weeks and the sample (Sharadar only)
# ─────────────────────────────────────────────────────────────────────────────

def weeks(period: str) -> list[date]:
    first, last, _ = PERIODS[period]
    by_week: dict[tuple, date] = {}
    for d in asof.trading_days(first, last):
        by_week[d.isocalendar()[:2]] = d
    return sorted(by_week.values())


def entry_day(a: date) -> date:
    return next(d for d in asof.nyse_schedule().index if d > a)


def dollar_volume(t: str, a: date) -> float | None:
    df = asof._raw(t)
    if df is None:
        return None
    w = df[df.index <= pd.Timestamp(a)].tail(DV_DAYS)
    if len(w) < DV_DAYS * 0.8:
        return None
    return float((w["close"] * w["volume"]).mean())


def spy_trend(d: date) -> str:
    """agent.Replay.spy_trend: SPY's prior close vs its 20-day average."""
    p = asof.prev_trading_day(d)
    spy = asof.split_adjusted("SPY", p - timedelta(days=90), p + timedelta(days=1))
    if spy.empty or len(spy) < 21:
        return "Neutral"
    c = spy["Close"]
    return "Bullish" if float(c.iloc[-1]) > float(c.rolling(20).mean().iloc[-1]) else "Bearish"


def build_universe() -> pd.DataFrame:
    rows, check = [], {}
    for period in PERIODS:
        ws = weeks(period)
        check[period] = {"weeks": len(ws), "first_asof": str(ws[0]), "last_asof": str(ws[-1])}
        for a in ws:
            d = entry_day(a)
            members = asof.members_live_format(a)
            dv = {t: dollar_volume(t, a) for t in members}
            ranked = sorted((t for t in members if dv[t]), key=lambda t: -dv[t])
            top = ranked[:TOP_N]
            rest = sorted(set(members) - set(top))
            rng = np.random.default_rng(int(a.strftime("%Y%m%d")))
            rnd = sorted(rng.choice(rest, size=min(RANDOM_N, len(rest)), replace=False).tolist())
            trend = spy_trend(d)
            for t in top:
                rows.append({"period": period, "asof": a.isoformat(), "entry_date": d.isoformat(), "ticker": t,
                             "group": "liquid", "dv_rank": ranked.index(t) + 1,
                             "dollar_volume_m": round(dv[t] / 1e6, 1), "spy_trend": trend})
            for t in rnd:
                rows.append({"period": period, "asof": a.isoformat(), "entry_date": d.isoformat(), "ticker": t,
                             "group": "random", "dv_rank": ranked.index(t) + 1 if t in ranked else None,
                             "dollar_volume_m": round(dv[t] / 1e6, 1) if dv.get(t) else None,
                             "spy_trend": trend})
            check[period].setdefault("members_without_prices", set()).update(t for t in members if not dv[t])
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIVERSE, index=False)
    for p in check.values():
        p["members_without_prices"] = sorted(p.get("members_without_prices", []))
    p = pd.read_parquet(config.SHARADAR_DIR / "prices.parquet", columns=["date"])
    check["sharadar_last_price_date"] = str(pd.to_datetime(p["date"]).max().date())
    try:
        for ds in [config.OPTIONS_DATASET, *config.EQUITY_DATASETS]:
            r = dbn.client().metadata.get_dataset_range(ds)
            check[f"databento_{ds}"] = {k: str(v) for k, v in (r.items() if isinstance(r, dict) else
                                                                [("start", r.start), ("end", r.end)])
                                       if k in ("start", "end")}
    except Exception as e:
        check["databento_range_error"] = f"{type(e).__name__}: {e}"[:300]
    (OUT / "data_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2)[:3000], flush=True)
    for period in PERIODS:
        u = df[df["period"] == period]
        print(f"::notice title=OOS universe {period}::{u['asof'].nunique()} weeks | {len(u)} stock-weeks | "
              f"{u['ticker'].nunique()} tickers | top liquid: "
              f"{' '.join(u[u['group'] == 'liquid']['ticker'].value_counts().index[:15])}", flush=True)
    print(f"::notice title=OOS data check::Sharadar prices to {check['sharadar_last_price_date']} | "
          + " | ".join(f"{k}: {v}" for k, v in check.items() if k.startswith("databento")), flush=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Sell rules on 1-minute quotes
# ─────────────────────────────────────────────────────────────────────────────

def simulate_1m(p: pd.DataFrame, cost: float, expiry: date, entry: date, *, sl=None, ceiling=None,
                trail=None, tday=None, tp=None) -> float:
    """% return of one rule set. Triggers on the mid, fills at the bid; a take-profit is a
    resting limit that fills at the target once the bid reaches it. Every rule also has
    the V3 force exit (13:00 ET on/after the day before expiry)."""
    ix = {d: i for i, d in enumerate(asof.nyse_schedule().index)}
    force_day = expiry - timedelta(days=1)
    ts = list(p["ts_et"]); bid = p["bid"].to_numpy(float); mid = p["mid"].to_numpy(float)
    peak = -math.inf
    for i in range(len(ts)):
        if tp is not None and bid[i] >= cost * (1 + tp / 100):
            return float(tp)
        pnl = (mid[i] - cost) / cost * 100
        peak = max(peak, pnl)
        t = ts[i]
        last_of_day = i == len(ts) - 1 or ts[i + 1].date() != t.date()
        out = (t.date() >= force_day and t.hour >= config.EXIT_FORCE_HOUR_ET)
        out |= sl is not None and pnl <= sl
        out |= ceiling is not None and pnl >= ceiling
        out |= trail is not None and peak >= trail[0] and pnl <= peak - trail[1]
        out |= tday is not None and ix[t.date()] - ix[entry] >= tday and last_of_day
        if out:
            return float((bid[i] - cost) / cost * 100)
    return float((bid[-1] - cost) / cost * 100)


V3 = dict(sl=config.EXIT_STOP_LOSS_PCT, ceiling=config.EXIT_HARD_CEILING_PCT,
          trail=(config.EXIT_TRAIL_TRIGGER_PCT, config.EXIT_TRAIL_GIVEBACK_PCT))


# ─────────────────────────────────────────────────────────────────────────────
#  One week of option data
# ─────────────────────────────────────────────────────────────────────────────

def _quotes(syms: list[str], start: pd.Timestamp, end: pd.Timestamp, purpose: str) -> pd.DataFrame:
    frames = []
    for i in range(0, len(syms), QUOTE_CHUNK):
        q = dbn.option_quotes(syms[i:i + QUOTE_CHUNK], start, end, purpose=purpose)
        if len(q):
            frames.append(q)
    q = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "ts_et", "bid", "ask"])
    if "symbol" not in q.columns and "raw_symbol" in q.columns:
        q = q.rename(columns={"raw_symbol": "symbol"})
    return q


def run_week(u: pd.DataFrame) -> list[dict]:
    a = date.fromisoformat(u["asof"].iloc[0])
    d = date.fromisoformat(u["entry_date"].iloc[0])
    entry = pd.Timestamp(datetime.combine(d, ENTRY))
    tickers = u["ticker"].tolist()
    dbn.prefetch_equity_day(d, tickers, "oos_bars")
    defs = {}
    for i in range(0, len(tickers), DEF_CHUNK):
        defs.update(defs_for_day(d, tickers[i:i + DEF_CHUNK], purpose="oos_definitions"))
    if dbn.QUOTE_ONLY:
        print(f"[oos] {a}: quoted bars + definitions for {len(tickers)} names | "
              f"quoted so far ${dbn.ledger().get('quoted_usd', 0):.2f}", flush=True)
        return []

    rows = []
    for r in u.to_dict("records"):
        base = {k: r[k] for k in ("period", "asof", "entry_date", "ticker", "group", "dv_rank", "spy_trend")}
        S = stock_at(r["ticker"], entry)
        if not S:
            rows.append({**base, "skip": "no stock price"}); continue
        base["stock_at_entry"] = S
        for side in ("CALL", "PUT"):
            o, why = pick(defs.get(r["ticker"], pd.DataFrame()), S, d, side == "CALL")
            if o is None:
                rows.append({**base, "side": side, "skip": why}); continue
            tds = asof.trading_days(d, o["expiry"])
            end_day = [x for x in tds if x < o["expiry"]][-1]
            rows.append({**base, "side": side, **o, "dte": (o["expiry"] - d).days, "end_day": end_day})

    # 1) entry snapshot for every picked contract
    picked = [x for x in rows if "raw_symbol" in x and not x.get("skip")]
    open_live = pd.Timestamp(asof.nyse_schedule().loc[d]["open_et"]) + ONE_MIN
    snap = _quotes(sorted({x["raw_symbol"] for x in picked}), pd.Timestamp(f"{d} 09:30"),
                   pd.Timestamp(f"{d} 09:47"), "oos_snapshot")
    for x in picked:
        s = snap[(snap["symbol"] == x["raw_symbol"]) & (snap["bid"] > 0) & (snap["ask"] > 0)
                 & (snap["ts_et"] >= open_live) & (snap["ts_et"] <= entry)]
        if s.empty:
            x["skip"] = "no live quote by 9:46"; continue
        bid, ask = float(s["bid"].iloc[-1]), float(s["ask"].iloc[-1])
        mid = (bid + ask) / 2
        T = max((pd.Timestamp(f"{x['expiry']} 16:00") - entry).total_seconds() / (365 * 86400), 1 / 365)
        call = x["side"] == "CALL"
        iv = implied_vol(mid, x["stock_at_entry"], x["strike"], T, config.RISK_FREE, call)
        dl = bs_delta(x["stock_at_entry"], x["strike"], T, config.RISK_FREE, iv, call) if iv else None
        sp = (ask - bid) / mid * 100
        x.update({"bid": bid, "ask": ask, "mid": mid, "spread_pct": sp, "iv_pct": iv * 100 if iv else None,
                  "delta": dl,
                  "qualifies": bool(dl is not None and dl >= MIN_DELTA and sp < SPREAD_MAX_PCT),
                  "tradable": bool(dl is not None and dl >= MIN_DELTA and sp <= TRADABLE_SPREAD_PCT)})

    # 2) full 1-minute path for every agent-tradable one (the qualifying ones + the 5-20% comparison)
    trad = [x for x in picked if x.get("tradable")]
    for grp_end in sorted({x["end_day"] for x in trad}):
        part = [x for x in trad if x["end_day"] == grp_end]
        q = _quotes(sorted({x["raw_symbol"] for x in part}), pd.Timestamp(f"{d} 09:30"),
                    pd.Timestamp(f"{grp_end} 16:00"), "oos_paths")
        for x in part:
            p = q[(q["symbol"] == x["raw_symbol"]) & (q["ts_et"] > entry)
                  & (q["ts_et"] <= pd.Timestamp(f"{grp_end} 16:00")) & (q["bid"] > 0) & (q["ask"] > 0)
                  & (q["ts_et"].dt.time >= time(9, 31)) & (q["ts_et"].dt.time <= time(16, 0))].copy()
            if p.empty:
                x["skip"] = "no quotes after entry"; continue
            p = p.sort_values("ts_et")
            p["mid"] = (p["bid"] + p["ask"]) / 2
            cost = x["ask"]
            x.update(v3_rule(p, cost, x["expiry"]))
            x.update({
                "v3_day2_pct": simulate_1m(p, cost, x["expiry"], d, tday=2, **V3),
                "day2_pct": simulate_1m(p, cost, x["expiry"], d, tday=2),
                "hold_pct": simulate_1m(p, cost, x["expiry"], d),
                "tp50_pct": simulate_1m(p, cost, x["expiry"], d, tp=50),
                "peak_pct": float(p["bid"].max() / cost - 1) * 100,
                "end_pct": float(p["bid"].iloc[-1] / cost - 1) * 100,
                "quote_minutes": int(len(p)),
            })
    n_q = sum(1 for x in trad if x.get("qualifies") and "v3_pct" in x)
    print(f"[oos] {a} entry {d}: {len(u)} names, {len(picked)} contracts, {len(trad)} tradable, "
          f"{n_q} qualifying measured | databento ${dbn.ledger()['spent_usd']:.2f}", flush=True)
    for x in rows:
        for k in ("expiry", "end_day"):
            if k in x and not isinstance(x[k], str):
                x[k] = x[k].isoformat()
    return rows


def shard(i: int, n: int) -> None:
    u = pd.read_csv(UNIVERSE)
    keys = sorted(u["asof"].unique())[i::n]
    base = dbn.ledger()
    rows = []
    try:
        for a in keys:
            rows += run_week(u[u["asof"] == a])
    except dbn.BudgetExceeded as e:
        print(f"::warning title=OOS shard {i} stopped on budget::{e}", flush=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PARTS / f"oos_{i:02d}.csv", index=False)
    now = dbn.ledger()
    delta = {k: now.get(k, 0) - base.get(k, 0) for k in ("spent_usd", "quoted_usd", "requests")}
    delta["by_purpose"] = {k: v - base.get("by_purpose", {}).get(k, 0) for k, v in now.get("by_purpose", {}).items()}
    (PARTS / f"spend_{i:02d}.json").write_text(json.dumps(delta))
    print(f"::notice title=OOS shard {i}::{len(keys)} weeks | {len(rows)} rows | spent ${delta['spent_usd']:.2f} | "
          f"quoted ${delta['quoted_usd']:.2f}", flush=True)


def merge() -> None:
    frames = []
    for f in sorted(PARTS.glob("oos_*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
    led = json.loads(dbn._LEDGER_REPO.read_text())
    spent = quoted = 0.0
    for f in sorted(PARTS.glob("spend_*.json")):
        dl = json.loads(f.read_text())
        spent += dl.get("spent_usd", 0); quoted += dl.get("quoted_usd", 0)
        led["requests"] = led.get("requests", 0) + dl.get("requests", 0)
        bp = led.setdefault("by_purpose", {})
        for k, v in dl.get("by_purpose", {}).items():
            if v:
                bp[k] = round(bp.get(k, 0) + v, 6)
    led["spent_usd"] = round(led.get("spent_usd", 0) + spent, 6)
    dbn._LEDGER_REPO.write_text(json.dumps(led, indent=2, sort_keys=True))
    if quoted and not spent:
        print(f"::notice title=OOS cost quote::all weeks, bars + option definitions: ${quoted:.2f} "
              f"(option quotes not included) | already spent ${led['spent_usd']:.2f}", flush=True)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        if TRADES.exists():                      # keep weeks measured by an earlier run
            old = pd.read_csv(TRADES)
            df = pd.concat([old[~old["asof"].isin(set(df["asof"]))], df], ignore_index=True)
        df = df.sort_values(["period", "asof", "group", "ticker", "side"])
        df.to_csv(TRADES, index=False)
        q = df[df["qualifies"] == True] if "qualifies" in df.columns else df.iloc[0:0]  # noqa: E712
        print(f"::notice title=OOS merged::{df['asof'].nunique()} weeks | {len(df)} rows | "
              f"{int(df['v3_pct'].notna().sum()) if 'v3_pct' in df else 0} paths | "
              f"{len(q)} qualifying | Databento spent ${led['spent_usd']:.2f}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring (pre-registered)
# ─────────────────────────────────────────────────────────────────────────────

def week_boot(x: pd.DataFrame, col: str, seed: int = BOOT_SEED) -> list[float]:
    """90% interval of the mean, resampling whole weeks."""
    wk = x.groupby("asof")[col].agg(["sum", "count"])
    S, C = wk["sum"].to_numpy(), wk["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wk), (BOOT, len(wk)))
    m = S[idx].sum(1) / C[idx].sum(1)
    return [round(float(v), 2) for v in np.percentile(m, [5, 95])]


def score_period(x: pd.DataFrame, split: str) -> dict:
    v = x["v3_pct"].astype(float)
    out = {"n": int(len(x)), "weeks": int(x["asof"].nunique()), "tickers": int(x["ticker"].nunique())}
    if len(x) < MIN_TRADES:
        return {**out, "verdict": "underpowered"}
    ci = week_boot(x, "v3_pct")
    h1, h2 = x[x["asof"] < split], x[x["asof"] >= split]
    k = max(1, math.ceil(DROP_TOP_SHARE * len(x)))
    drop_top = float(v.sort_values().iloc[:-k].mean())
    month = x["entry_date"].str[:7]
    best_month = x.groupby(month)["v3_pct"].sum().idxmax()
    no_best = float(x.loc[month != best_month, "v3_pct"].mean())
    c = {
        "1_mean_and_ci_above_zero": bool(v.mean() > 0 and ci[0] > 0),
        "2_both_halves_positive": bool(h1["v3_pct"].mean() > 0 and h2["v3_pct"].mean() > 0),
        "3_positive_without_top_3pct": bool(drop_top > 0),
        "4_positive_without_best_month": bool(no_best > 0),
    }
    return {**out, "mean": round(float(v.mean()), 2), "median": round(float(v.median()), 2),
            "win_rate": round(float((v > 0).mean() * 100), 1), "ci90": ci,
            "h1": {"n": int(len(h1)), "mean": round(float(h1["v3_pct"].mean()), 2)},
            "h2": {"n": int(len(h2)), "mean": round(float(h2["v3_pct"].mean()), 2)},
            "drop_top": {"k": k, "mean": round(drop_top, 2)},
            "best_month": {"month": best_month, "mean_without": round(no_best, 2)},
            "criteria": c, "verdict": "pass" if all(c.values()) else "fail"}


def secondary(x: pd.DataFrame, allp: pd.DataFrame) -> dict:
    """Reported, not part of the pass/fail."""
    def m(s):
        s = s.dropna()
        return {"n": int(len(s)), "mean": round(float(s.mean()), 2) if len(s) else None}
    wide = allp[(allp["tradable"] == True) & (allp["spread_pct"] >= SPREAD_MAX_PCT)]  # noqa: E712
    one_side = x[((x["spy_trend"] == "Bullish") & (x["side"] == "CALL")) |
                 ((x["spy_trend"] != "Bullish") & (x["side"] == "PUT"))]
    meas = allp[allp["v3_pct"].notna() & (allp["delta"] >= MIN_DELTA)]
    return {
        "rules_on_qualifying": {r: m(x[c]) for r, c in [("v3", "v3_pct"), ("v3_plus_day2", "v3_day2_pct"),
                                                         ("day2_only", "day2_pct"), ("hold_to_thursday", "hold_pct"),
                                                         ("take_profit_50", "tp50_pct")]},
        "wide_5_to_20_v3": m(wide["v3_pct"]),
        "by_group": {g: m(x.loc[x["group"] == g, "v3_pct"]) for g in ("liquid", "random")},
        "by_side": {s: m(x.loc[x["side"] == s, "v3_pct"]) for s in ("CALL", "PUT")},
        "one_side_follow_spy_trend": m(one_side["v3_pct"]),
        "cutoff_sensitivity": {f"spread<{c}": m(meas.loc[meas["spread_pct"] < c, "v3_pct"]) for c in (3, 4, 5, 6, 8, 10)},
        "top_tickers": x.groupby("ticker")["v3_pct"].agg(["count", "mean"]).round(1)
                        .sort_values("count", ascending=False).head(15).reset_index().to_dict("records"),
    }


def score() -> dict:
    df = pd.read_csv(TRADES)
    df["qualifies"] = df["qualifies"].fillna(False).astype(bool)
    df["tradable"] = df["tradable"].fillna(False).astype(bool)
    out = {"locked": {"top_n": TOP_N, "random_n": RANDOM_N, "spread_max_pct": SPREAD_MAX_PCT,
                      "min_delta": MIN_DELTA, "min_trades": MIN_TRADES, "drop_top_share": DROP_TOP_SHARE},
           "periods": {}}
    for period, (_, _, split) in PERIODS.items():
        allp = df[df["period"].astype(str) == period]
        x = allp[allp["qualifies"] & allp["v3_pct"].notna()]
        out["periods"][period] = {**score_period(x, split), "secondary": secondary(x, allp),
                                  "skips": allp["skip"].value_counts().to_dict() if "skip" in allp else {}}
    v = [p["verdict"] for p in out["periods"].values()]
    out["verdict"] = ("confirmed" if v.count("pass") == len(v) else
                      "underpowered" if "underpowered" in v else
                      "not confirmed" if "pass" in v else "rejected")
    VERDICT.write_text(json.dumps(out, indent=2, default=str))
    for period, p in out["periods"].items():
        print(f"::notice title=OOS {period}: {p['verdict']}::n {p['n']} | mean {p.get('mean')}% | CI90 {p.get('ci90')} | "
              f"H1 {p.get('h1')} | H2 {p.get('h2')} | criteria {p.get('criteria')}", flush=True)
    print(f"::notice title=OOS verdict::{out['verdict']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.universe:
        build_universe()
    elif a.merge:
        merge()
    elif a.score:
        score()
    else:
        shard(a.shard or 0, a.nshards)


if __name__ == "__main__":
    main()
