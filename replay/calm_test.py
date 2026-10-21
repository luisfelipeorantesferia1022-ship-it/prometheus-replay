"""
Do calmer, lower-error stocks give bigger option peaks?  Tested on real OPRA
quotes (the stage-2 backtest prices options with Black-Scholes, so it can't
answer this without assuming the answer).

Sample, for each of the 52 weekly as-of dates:
  * every stage-1 candidate (the names the funnel actually draws from), and
  * 10 other S&P 500 members picked at random (seeded), so the test also
    covers stocks the LSTM did not single out.

For each name, on the first trading day of the following week (when that
week's whitelist goes live) at 9:46 ET, it "buys" BOTH the call and the put
the agent's own rule would pick (12-24 DTE, 14-21 preferred; strike nearest
1% out of the money), at the ask, so the LSTM's direction call plays no part.
Then it follows the real bid every minute until the agent's Thursday-before-
expiry exit and records:
  peak_pct   best bid / ask - 1          (the best moment to sell, hindsight)
  peak5_pct  the same within 5 trading days
  end_pct    bid at the Thursday exit / ask - 1
along with the stock's stage-1 LSTM error (MAE), hit rate, direction accuracy,
1-year and 20-day volatility, beta, and the option's implied vol and spread.

    python -m replay.calm_test [--weeks N]      # results/analysis/calm_test.csv
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from . import asof, config, dbn
from .agent import bs_delta, implied_vol
from .trade_factors import alpha_beta

R = config.RESULTS_DIR
OUT = R / "analysis" / "calm_test.csv"
CONTROLS_PER_WEEK = 10
ENTRY = time(9, 46)
ONE_MIN = pd.Timedelta(minutes=1)


def sample(a: date) -> list[tuple[str, str]]:
    wk = R / "weekly" / a.isoformat()
    s1 = json.loads((wk / "stage1.json").read_text())
    cands = list(dict.fromkeys(s1.get("candidates", [])))
    acc = pd.read_csv(wk / "lstm_accuracy_results.csv")
    others = sorted(set(acc["ticker"]) - set(cands))
    rng = np.random.default_rng(int(a.strftime("%Y%m%d")))
    ctrl = list(rng.choice(others, size=min(CONTROLS_PER_WEEK, len(others)), replace=False))
    return [(t, "candidate") for t in cands] + [(t, "control") for t in ctrl]


def defs_for_day(d: date, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Option definitions for many underlyings with one request, split into
    the per-ticker cache files dbn.option_definitions reads."""
    def path(t):
        return config.DBN_CACHE_DIR / "opra_def" / d.isoformat() / f"{dbn._db_sym(t)}.parquet"
    missing = [t for t in tickers if not path(t).exists()]
    if missing:
        s = pd.Timestamp(f"{d} 00:00", tz="UTC")
        cache = config.DBN_CACHE_DIR / "opra_def_batch" / f"{d}_{dbn._hash(missing)}.parquet"
        raw = dbn._get("calm_definitions", cache, dataset=config.OPTIONS_DATASET, schema="definition",
                       symbols=[f"{dbn._db_sym(t)}.OPT" for t in missing], stype_in="parent",
                       start=s, end=s + pd.Timedelta(hours=14))
        if raw is not None and not raw.empty:
            key = raw["underlying"].astype(str).str.strip() if "underlying" in raw.columns else None
            for t in missing:
                sub = raw[key == dbn._db_sym(t)] if key is not None else raw.iloc[0:0]
                if len(sub):
                    path(t).parent.mkdir(parents=True, exist_ok=True)
                    sub.to_parquet(path(t))
    out = {}
    for t in tickers:
        try:
            out[t] = dbn.option_definitions(t, d)     # cached, or a single request as fallback
        except dbn.BudgetExceeded:
            raise
        except Exception as e:
            print(f"[calm] defs {t} {d}: {type(e).__name__}: {e}", flush=True)
            out[t] = pd.DataFrame()
    return out


def pick(defs: pd.DataFrame, S: float, d: date, call: bool):
    if defs.empty:
        return None, "no definitions"
    root = defs["raw_symbol"].str[:6].str.strip()
    defs = defs[~root.str.contains(r"\d")]              # skip adjusted / non-standard contracts
    now = datetime.combine(d, ENTRY)
    target = fallback = None
    for exp in sorted(defs["expiration"].unique()):
        days_out = (datetime.combine(exp, datetime.min.time()) - now).days
        if days_out < 12 or days_out > 24:
            continue
        if fallback is None:
            fallback = exp
        if 14 <= days_out <= 21:
            target = exp
            break
    target = target or fallback
    if target is None:
        return None, "no expiry in 12-24 DTE"
    o = defs[(defs["expiration"] == target) & (defs["instrument_class"] == ("C" if call else "P"))].copy()
    o["strike"] = o["strike_price"].astype(float)
    lo, hi = (S, S * 1.03) if call else (S * 0.97, S * 1.001)
    v = o[(o["strike"] >= lo) & (o["strike"] <= hi)]
    if v.empty:
        v = o[(o["strike"] >= S) & (o["strike"] <= S * 1.05)] if call else \
            o[(o["strike"] >= S * 0.95) & (o["strike"] <= S * 1.001)]
    if v.empty:
        return None, "no strike in range"
    ideal = S * 1.01 if call else S * 0.99
    r = v.assign(_d=(v["strike"] - ideal).abs()).sort_values("_d", kind="stable").iloc[0]
    return {"raw_symbol": r["raw_symbol"], "strike": float(r["strike"]), "expiry": target}, None


def stock_at(t: str, ts: pd.Timestamp) -> float | None:
    bars = dbn.equity_bars(t, ts.date(), purpose="calm_bars")
    if bars.empty:
        return None
    done = bars[bars.index + ONE_MIN <= ts]
    return float(done["close"].iloc[-1]) if not done.empty else None


def moves(t: str, d: date, entry: pd.Timestamp, end: date, S: float) -> dict:
    hi, lo = [], []
    bars = dbn.equity_bars(t, d, purpose="calm_bars")
    after = bars[bars.index >= entry] if not bars.empty else bars
    if len(after):
        hi.append(after["high"].max()); lo.append(after["low"].min())
    df = asof._raw(t)
    if df is not None:
        w = df[(df.index > pd.Timestamp(d)) & (df.index <= pd.Timestamp(end))]
        k = w["closeunadj"] / w["close"]
        hi += (w["high"] * k).tolist(); lo += (w["low"] * k).tolist()
    if not hi:
        return {}
    return {"max_up_pct": (max(hi) / S - 1) * 100, "max_down_pct": (1 - min(lo) / S) * 100}


def run_week(a: date) -> list[dict]:
    days = asof.week_trading_days(a)
    if not days:
        return []
    d = days[0]
    entry = pd.Timestamp(datetime.combine(d, ENTRY))
    names = sample(a)
    tickers = [t for t, _ in names]
    acc = pd.read_csv(R / "weekly" / a.isoformat() / "lstm_accuracy_results.csv").set_index("ticker")
    acc["mae_pctile_week"] = acc["avg_mae_pct"].rank(pct=True) * 100
    dbn.prefetch_equity_day(d, tickers, "calm_bars")
    defs = defs_for_day(d, tickers)

    rows, chosen = [], []
    for t, grp in names:
        base = {"asof": a.isoformat(), "entry_date": d.isoformat(), "ticker": t, "group": grp}
        if t in acc.index:
            r = acc.loc[t]
            base.update({k: r[k] for k in ["avg_mae_pct", "hit_rate_pct", "dir_accuracy_pct", "avg_snr",
                                            "mae_pctile_week"]})
        S = stock_at(t, entry)
        if not S:
            rows.append({**base, "skip": "no stock price"}); continue
        base["stock_at_entry"] = S
        try:
            base.update({k: v for k, v in alpha_beta(t, d, 1).items()
                         if k in ("beta_1y", "vol_1y_pct", "vol_20d_pct", "idio_vol_1y_pct")})
        except Exception as e:
            print(f"[calm] alpha/beta {t}: {type(e).__name__}", flush=True)
        for side in ("CALL", "PUT"):
            o, why = pick(defs.get(t, pd.DataFrame()), S, d, side == "CALL")
            if o is None:
                rows.append({**base, "side": side, "skip": why}); continue
            tds = asof.trading_days(d, o["expiry"])
            end_day = [x for x in tds if x < o["expiry"]][-1]
            row = {**base, "side": side, **o, "dte": (o["expiry"] - d).days,
                   "end_day": end_day.isoformat(),
                   "day5": tds[min(4, len(tds) - 1)].isoformat()}
            row.update(moves(t, d, entry, end_day, S))
            rows.append(row); chosen.append(row)

    if chosen:
        last = max(pd.Timestamp(f"{r['end_day']} 16:00") for r in chosen)
        q = dbn.option_quotes([r["raw_symbol"] for r in chosen], pd.Timestamp(f"{d} 09:30"), last,
                              purpose="calm_quotes")
        col = "symbol" if "symbol" in q.columns else "raw_symbol"
        open_live = pd.Timestamp(asof.nyse_schedule().loc[d]["open_et"]) + ONE_MIN
        for r in chosen:
            p = q[q[col] == r["raw_symbol"]] if len(q) else q
            if p.empty:
                r["skip"] = "no quotes"; continue
            p = p[(p["bid"] > 0) & (p["ask"] > 0)]
            first = p[(p["ts_et"] >= open_live) & (p["ts_et"] <= entry)]
            if first.empty:
                r["skip"] = "no live quote by 9:46"; continue
            e = first.iloc[-1]
            bid, ask = float(e["bid"]), float(e["ask"])
            mid = (bid + ask) / 2
            T = max((pd.Timestamp(f"{r['expiry']} 16:00") - entry).total_seconds() / (365 * 86400), 1 / 365)
            call = r["side"] == "CALL"
            iv = implied_vol(mid, r["stock_at_entry"], r["strike"], T, config.RISK_FREE, call)
            r.update({"bid": bid, "ask": ask, "mid": mid, "spread_pct": (ask - bid) / mid * 100,
                      "iv_pct": iv * 100 if iv else None})
            if iv:
                dl = bs_delta(r["stock_at_entry"], r["strike"], T, config.RISK_FREE, iv, call)
                r.update({"delta": dl, "leverage": dl * r["stock_at_entry"] / mid})
            end_ts = pd.Timestamp(f"{r['end_day']} 16:00")
            path = p[(p["ts_et"] > entry) & (p["ts_et"] <= end_ts)
                     & (p["ts_et"].dt.time >= time(9, 31)) & (p["ts_et"].dt.time <= time(16, 0))]
            if path.empty:
                r["skip"] = "no quotes after entry"; continue
            pk = path.loc[path["bid"].idxmax()]
            p5 = path[path["ts_et"].dt.date <= date.fromisoformat(r["day5"])]
            r.update({"peak_bid": float(pk["bid"]), "peak_at": str(pk["ts_et"]),
                      "peak_pct": (float(pk["bid"]) / ask - 1) * 100,
                      "days_to_peak": int(np.busday_count(d, pd.Timestamp(pk["ts_et"]).date())),
                      "peak5_pct": (float(p5["bid"].max()) / ask - 1) * 100 if len(p5) else None,
                      "end_bid": float(path["bid"].iloc[-1]),
                      "end_pct": (float(path["bid"].iloc[-1]) / ask - 1) * 100,
                      "quote_minutes": int(len(path))})
    print(f"[calm] {a} entry {d}: {len(names)} names, {sum(1 for r in chosen if 'peak_pct' in r)} options "
          f"measured | databento ${dbn.ledger()['spent_usd']:.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=0, help="only the first N weeks (0 = all)")
    a = ap.parse_args()
    weeks = asof.weekly_asof_dates()
    if a.weeks:
        weeks = weeks[:a.weeks]
    rows = []
    try:
        for w in weeks:
            rows += run_week(w)
    except dbn.BudgetExceeded as e:
        print(f"::warning title=Calm test stopped on budget::{e}", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    ok = df["peak_pct"].notna().sum() if "peak_pct" in df.columns else 0
    skips = df["skip"].value_counts().to_dict() if "skip" in df.columns else {}
    print(f"::notice title=Calm test::{len(weeks)} weeks | {ok} options measured | skipped {skips} | "
          f"Databento spend ${dbn.ledger()['spent_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
