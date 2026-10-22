"""
Price paths for testing sell rules, saved compactly so the rules can be tried
locally without buying data again.

1. Calm-test options (1,518, both sides, bought at the ask at 9:46): every
   5-minute bucket from entry to the Thursday exit day's close, with the bid's
   open/high/low/close and the mid's high/low/close. Also, at full 1-minute
   precision, what the agent's current V3 sell rules would have done (trigger on
   the mid, sell at the bid): stop -75%, ceiling +200%, trail 20 points once up
   50%, force-exit 13:00 the day before expiry.
       -> results/analysis/exit_paths.parquet, exit_v3_check.csv
2. The agent's own trades (results/analysis/trade_factors.csv): 1-minute bid,
   ask and stock price from entry to expiry, for charts.
       -> results/analysis/agent_trade_paths.parquet

    python -m replay.exit_paths --shard i --nshards n   (calm options, one shard of weeks)
    python -m replay.exit_paths --merge
    python -m replay.exit_paths --agent
"""

from __future__ import annotations

import argparse
import json
from datetime import date, time, timedelta

import numpy as np
import pandas as pd

from . import asof, config, dbn

R = config.RESULTS_DIR / "analysis"
PARTS = R / "exit_parts"
ENTRY = time(9, 46)
CHUNK = 8


def option_key(r) -> str:
    return f"{r['raw_symbol']}|{r['entry_date']}"


def v3_rule(p: pd.DataFrame, cost: float, expiry: date) -> dict:
    """The live V3 exit engine on 1-minute quotes (real mode)."""
    force_day = expiry - timedelta(days=1)
    peak = None
    for ts, bid, mid in zip(p["ts_et"], p["bid"], p["mid"]):
        pnl = (mid - cost) / cost * 100
        peak = pnl if peak is None else max(peak, pnl)
        force = ts.date() >= force_day and ts.hour >= config.EXIT_FORCE_HOUR_ET
        reason = None
        if pnl <= config.EXIT_STOP_LOSS_PCT:
            reason = "Stop loss"
        elif pnl >= config.EXIT_HARD_CEILING_PCT:
            reason = "Hard ceiling"
        elif force:
            reason = "Force exit"
        elif peak >= config.EXIT_TRAIL_TRIGGER_PCT and pnl <= peak - config.EXIT_TRAIL_GIVEBACK_PCT:
            reason = "Trail stop"
        if reason:
            return {"v3_exit_ts": str(ts), "v3_reason": reason, "v3_pct": (bid - cost) / cost * 100}
    last = p.iloc[-1]
    return {"v3_exit_ts": str(last["ts_et"]), "v3_reason": "End of data",
            "v3_pct": (float(last["bid"]) - cost) / cost * 100}


def run_week(a: str, rows: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict]]:
    d = date.fromisoformat(rows["entry_date"].iloc[0])
    entry = pd.Timestamp(f"{d} 09:46")
    recs = rows.to_dict("records")
    frames = []
    for i in range(0, len(recs), CHUNK):
        part = recs[i:i + CHUNK]
        last = max(pd.Timestamp(f"{r['end_day']} 16:00") for r in part)
        q = dbn.option_quotes([r["raw_symbol"] for r in part], pd.Timestamp(f"{d} 09:30"), last,
                              purpose="exit_paths_quotes")
        if len(q):
            frames.append(q)
    q = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    col = "symbol" if "symbol" in q.columns else "raw_symbol"
    paths, checks = [], []
    for r in recs:
        p = q[q[col] == r["raw_symbol"]] if len(q) else q
        if p.empty:
            continue
        end_ts = pd.Timestamp(f"{r['end_day']} 16:00")
        p = p[(p["ts_et"] > entry) & (p["ts_et"] <= end_ts) & (p["bid"] > 0) & (p["ask"] > 0)
              & (p["ts_et"].dt.time >= time(9, 31)) & (p["ts_et"].dt.time <= time(16, 0))].copy()
        if p.empty:
            continue
        p["mid"] = (p["bid"] + p["ask"]) / 2
        cost = float(r["ask"])
        checks.append({"key": option_key(r), **v3_rule(p, cost, date.fromisoformat(r["expiry"]))})
        g = p.groupby(p["ts_et"].dt.floor("5min"))
        b = pd.DataFrame({"bid_o": g["bid"].first(), "bid_h": g["bid"].max(), "bid_l": g["bid"].min(),
                          "bid_c": g["bid"].last(), "mid_h": g["mid"].max(), "mid_l": g["mid"].min(),
                          "mid_c": g["mid"].last()}).astype("float32").reset_index()
        b.insert(0, "key", option_key(r))
        paths.append(b)
    print(f"[exit] {a}: {len(paths)} paths | databento ${dbn.ledger()['spent_usd']:.2f}", flush=True)
    return paths, checks


def shard(i: int, n: int) -> None:
    d = pd.read_csv(R / "calm_test.csv")
    d = d[d["peak_pct"].notna()]
    weeks = sorted(d["asof"].unique())[i::n]
    base = dbn.ledger()
    paths, checks = [], []
    for a in weeks:
        p, c = run_week(a, d[d["asof"] == a])
        paths += p; checks += c
    PARTS.mkdir(parents=True, exist_ok=True)
    if paths:
        pd.concat(paths, ignore_index=True).to_parquet(PARTS / f"paths_{i:02d}.parquet", index=False)
    pd.DataFrame(checks).to_csv(PARTS / f"v3_{i:02d}.csv", index=False)
    now = dbn.ledger()
    (PARTS / f"spend_{i:02d}.json").write_text(json.dumps({
        "spent_usd": now["spent_usd"] - base.get("spent_usd", 0),
        "requests": now.get("requests", 0) - base.get("requests", 0),
        "by_purpose": {k: v - base.get("by_purpose", {}).get(k, 0) for k, v in now.get("by_purpose", {}).items()}}))
    print(f"::notice title=Exit paths shard {i}::{len(weeks)} weeks | {len(checks)} options", flush=True)


def merge() -> None:
    paths = [pd.read_parquet(f) for f in sorted(PARTS.glob("paths_*.parquet"))]
    pd.concat(paths, ignore_index=True).to_parquet(R / "exit_paths.parquet", index=False,
                                                   compression="zstd")
    v3 = [pd.read_csv(f) for f in sorted(PARTS.glob("v3_*.csv")) if f.stat().st_size > 5]
    pd.concat(v3, ignore_index=True).to_csv(R / "exit_v3_check.csv", index=False)
    led = json.loads(dbn._LEDGER_REPO.read_text())
    for f in sorted(PARTS.glob("spend_*.json")):
        dl = json.loads(f.read_text())
        led["spent_usd"] = round(led.get("spent_usd", 0) + dl["spent_usd"], 6)
        led["requests"] = led.get("requests", 0) + dl.get("requests", 0)
        bp = led.setdefault("by_purpose", {})
        for k, v in dl.get("by_purpose", {}).items():
            if v:
                bp[k] = round(bp.get(k, 0) + v, 6)
    dbn._LEDGER_REPO.write_text(json.dumps(led, indent=2, sort_keys=True))
    n = pd.read_parquet(R / "exit_paths.parquet", columns=["key"])["key"].nunique()
    size = (R / "exit_paths.parquet").stat().st_size / 1e6
    print(f"::notice title=Exit paths merged::{n} options | {size:.1f} MB | Databento ${led['spent_usd']:.2f}",
          flush=True)


def agent_paths() -> None:
    t = pd.read_csv(R / "trade_factors.csv")
    out = []
    for _, r in t.iterrows():
        entry = pd.Timestamp(r["entry_ts"])
        exp = pd.Timestamp(r["expiry"]).date()
        sym = None
        bk = r["seen_in"].split()[0].split("/")
        tr = pd.read_csv(config.RESULTS_DIR / "agent" / bk[0] / bk[1] / "trades.csv")
        tr = tr[(tr["ticker"] == r["ticker"]) & (pd.to_datetime(tr["entry_ts"]).dt.floor("s") == entry.floor("s"))]
        if len(tr):
            sym = tr["raw_symbol"].iloc[0]
        if not sym:
            continue
        q = dbn.option_quotes([sym], pd.Timestamp(f"{entry.date()} 09:00"), pd.Timestamp(f"{exp} 16:15"),
                              purpose="analysis_quotes")
        if q.empty:
            continue
        q = q[(q["ts_et"] >= entry.floor("min")) & (q["ts_et"].dt.time >= time(9, 31))
              & (q["ts_et"].dt.time <= time(16, 0)) & (q["bid"] > 0) & (q["ask"] > 0)]
        stock = []
        for day in asof.trading_days(entry.date(), exp):
            b = dbn.equity_bars(r["ticker"], day, purpose="agent_path_bars")
            if not b.empty:
                s = b["close"].copy()
                s.index = s.index + pd.Timedelta(minutes=1)        # price known at bar end
                stock.append(s)
        s = pd.concat(stock).sort_index() if stock else pd.Series(dtype=float)
        p = q[["ts_et", "bid", "ask"]].copy()
        p["stock"] = s.reindex(p["ts_et"], method="ffill").values if len(s) else np.nan
        p.insert(0, "key", f"{sym}|{entry}")
        out.append(p)
    df = pd.concat(out, ignore_index=True)
    df[["bid", "ask", "stock"]] = df[["bid", "ask", "stock"]].astype("float32")
    df.to_parquet(R / "agent_trade_paths.parquet", index=False, compression="zstd")
    print(f"::notice title=Agent trade paths::{df['key'].nunique()} trades, {len(df)} minutes | "
          f"Databento ${dbn.ledger()['spent_usd']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--agent", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.agent:
        agent_paths()
    else:
        shard(a.shard or 0, a.nshards)


if __name__ == "__main__":
    main()
