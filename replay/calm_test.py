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
QUOTE_CHUNK = 8          # option symbols per quote request


def sample(a: date) -> list[tuple[str, str]]:
    wk = R / "weekly" / a.isoformat()
    s1 = json.loads((wk / "stage1.json").read_text())
    cands = list(dict.fromkeys(s1.get("candidates", [])))
    acc = pd.read_csv(wk / "lstm_accuracy_results.csv")
    others = sorted(set(acc["ticker"]) - set(cands))
    rng = np.random.default_rng(int(a.strftime("%Y%m%d")))
    ctrl = list(rng.choice(others, size=min(CONTROLS_PER_WEEK, len(others)), replace=False))
    return [(t, "candidate") for t in cands] + [(t, "control") for t in ctrl]


def opt_root(t: str) -> str:
    """OPRA root: class shares drop the dot (BF.B -> BFB, BRK.B -> BRKB)."""
    return t.replace("-", "").replace(".", "")


def defs_for_day(d: date, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Option definitions for many underlyings with one request, split by OSI
    root into the per-ticker cache files dbn.option_definitions reads."""
    def path(t):
        return config.DBN_CACHE_DIR / "opra_def" / d.isoformat() / f"{dbn._db_sym(t)}.parquet"
    missing = [t for t in tickers if not path(t).exists()]
    if missing:
        s = pd.Timestamp(f"{d} 00:00", tz="UTC")
        cache = config.DBN_CACHE_DIR / "opra_def_batch" / f"{d}_{dbn._hash(missing)}.parquet"
        raw = dbn._get("calm_definitions", cache, dataset=config.OPTIONS_DATASET, schema="definition",
                       symbols=[f"{opt_root(t)}.OPT" for t in missing], stype_in="parent",
                       start=s, end=s + pd.Timedelta(hours=14))
        if raw is not None and not raw.empty:
            root = raw["raw_symbol"].astype(str).str[:6].str.strip()
            for t in missing:
                sub = raw[root == opt_root(t)]
                if len(sub):
                    path(t).parent.mkdir(parents=True, exist_ok=True)
                    sub.to_parquet(path(t))
    out = {}
    for t in tickers:
        # Only read what the batch cached; never fall back to a per-ticker
        # request (its parent symbol is wrong for class shares).
        out[t] = dbn.option_definitions(t, d) if path(t).exists() else pd.DataFrame()
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
        # Several small requests: one large one kept getting cut off mid-stream.
        frames = []
        for i in range(0, len(chosen), QUOTE_CHUNK):
            part = chosen[i:i + QUOTE_CHUNK]
            last = max(pd.Timestamp(f"{r['end_day']} 16:00") for r in part)
            frames.append(dbn.option_quotes([r["raw_symbol"] for r in part], pd.Timestamp(f"{d} 09:30"),
                                            last, purpose="calm_quotes"))
        frames = [f for f in frames if len(f)]
        q = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        col = "symbol" if "symbol" in q.columns else "raw_symbol"
        if q.empty:
            q = pd.DataFrame(columns=[col, "ts_et", "bid", "ask"])
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
    ap.add_argument("--shard", type=int, default=None, help="run weeks shard::nshards only")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--skip-done", action="store_true", help="skip weeks already in calm_test.csv")
    ap.add_argument("--merge", action="store_true", help="combine shard parts into calm_test.csv")
    a = ap.parse_args()
    parts = R / "analysis" / "calm_parts"
    if a.merge:
        return merge(parts)
    weeks = asof.weekly_asof_dates()
    if a.weeks:
        weeks = weeks[:a.weeks]
    if a.shard is not None:
        weeks = weeks[a.shard::a.nshards]
    if a.skip_done and OUT.exists():
        done = set(pd.read_csv(OUT)["asof"])
        weeks = [w for w in weeks if w.isoformat() not in done]
    base = dbn.ledger()
    rows = []
    try:
        for w in weeks:
            rows += run_week(w)
    except dbn.BudgetExceeded as e:
        print(f"::warning title=Calm test stopped on budget::{e}", flush=True)
    df = pd.DataFrame(rows)
    out = OUT if a.shard is None else parts / f"calm_test_{a.shard:02d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if a.shard is not None:          # spend of this shard alone, added up by --merge
        now = dbn.ledger()
        delta = {"spent_usd": now["spent_usd"] - base.get("spent_usd", 0),
                 "requests": now.get("requests", 0) - base.get("requests", 0),
                 "by_purpose": {k: v - base.get("by_purpose", {}).get(k, 0)
                                for k, v in now.get("by_purpose", {}).items()}}
        (parts / f"spend_{a.shard:02d}.json").write_text(json.dumps(delta))
    ok = df["peak_pct"].notna().sum() if "peak_pct" in df.columns else 0
    skips = df["skip"].value_counts().to_dict() if "skip" in df.columns else {}
    print(f"::notice title=Calm test {'' if a.shard is None else f'shard {a.shard}'}::{len(weeks)} weeks | "
          f"{ok} options measured | skipped {skips} | Databento spend ${dbn.ledger()['spent_usd']:.2f}",
          flush=True)


def merge(parts) -> None:
    files = sorted(parts.glob("calm_test_*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["asof"])
    if OUT.exists():                 # keep weeks measured by an earlier run
        old = pd.read_csv(OUT)
        old = old[~old["asof"].isin(set(new["asof"]))]
        new = pd.concat([old, new], ignore_index=True)
    df = new.sort_values(["asof", "group", "ticker", "side"])
    df.to_csv(OUT, index=False)
    # Repo ledger + every shard's own spend.
    led = json.loads(dbn._LEDGER_REPO.read_text())
    for f in sorted(parts.glob("spend_*.json")):
        dlt = json.loads(f.read_text())
        led["spent_usd"] = round(led.get("spent_usd", 0) + dlt["spent_usd"], 6)
        led["requests"] = led.get("requests", 0) + dlt.get("requests", 0)
        bp = led.setdefault("by_purpose", {})
        for k, v in dlt.get("by_purpose", {}).items():
            if v:
                bp[k] = round(bp.get(k, 0) + v, 6)
    dbn._LEDGER_REPO.write_text(json.dumps(led, indent=2, sort_keys=True))
    ok = int(df["peak_pct"].notna().sum())
    print(f"::notice title=Calm test merged::{len(files)} shards | {df['asof'].nunique()} weeks | "
          f"{ok} options measured | skipped {df['skip'].value_counts().to_dict() if 'skip' in df.columns else {}} | "
          f"Databento spend ${led['spent_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
