"""
The live agent trains a fresh LSTM for every whitelisted ticker each entry
morning (lstm_forecast.run_lstm_forecast). The model only sees daily history
through the previous close, so its fit does not depend on the live quote; the
quote enters only in tom_pct / mae_pct / signal_to_noise, which the agent
replay recomputes from the stored raw outputs at the simulated scan time.

That makes the fits independent of the book, so they are precomputed here in
parallel shards.

    python -m replay.forecasts plan
    python -m replay.forecasts shard --shard 0 --nshards 8
    python -m replay.forecasts merge
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import pandas as pd

from . import asof, config, shims
from .weekly import whitelist_timeline

FORECASTS_CSV = config.RESULTS_DIR / "agent" / "lstm_forecasts.csv"


def earnings_soon(ticker: str, d: date, days: int = config.EARNINGS_SOON_DAYS) -> bool:
    """trading_agent_v3._has_earnings_soon: an earnings date in [d, d+days]."""
    cutoff = d + timedelta(days=days)
    return any(d <= e <= cutoff for e in asof.earnings_dates(ticker))


def jobs(include_done: bool = False) -> list[tuple[date, str]]:
    """
    Every (entry morning, ticker) any variant's agent could scan. Fits do not
    depend on the variant, so the union is fitted once; fits already in
    FORECASTS_CSV are skipped.
    """
    out = set()
    for variant in config.SIM_MIN_TRADES_VARIANTS:
        for a, wl in whitelist_timeline(variant).items():
            for d in asof.week_trading_days(a):
                if d.weekday() not in config.ENTRY_ALLOWED_WEEKDAYS:
                    continue
                for t in wl:
                    if not earnings_soon(t, d):
                        out.add((d, t))
    if not include_done and FORECASTS_CSV.exists():
        done = pd.read_csv(FORECASTS_CSV)
        out -= {(date.fromisoformat(str(r.date)), r.ticker) for r in done.itertuples()}
    return sorted(out)


def run_shard(shard: int, nshards: int) -> None:
    mine = [j for i, j in enumerate(jobs()) if i % nshards == shard]
    print(f"[forecasts] shard {shard}/{nshards}: {len(mine)} fits", flush=True)
    rows = []
    for i, (d, t) in enumerate(mine, 1):
        data_asof = asof.prev_trading_day(d)
        lf = shims.lstm_forecast_module(data_asof, tag=d.isoformat())
        lf._current = t
        last = asof.split_adjusted(t, data_asof - timedelta(days=10), data_asof + timedelta(days=1))
        if last.empty:
            rows.append({"date": d, "ticker": t, "ok": False, "reason": "no data"})
            continue
        last_close = float(last["Close"].iloc[-1])
        t0 = time.time()
        fc = lf.run_lstm_forecast(t, last_close)
        if fc is None:
            rows.append({"date": d, "ticker": t, "ok": False, "reason": "run_lstm_forecast None",
                         "last_close": last_close})
        else:
            rows.append({"date": d, "ticker": t, "ok": True, "last_close": last_close,
                         "tomorrow_price": fc["tomorrow_price"],
                         "fiveday_price": fc["fiveday_price"],
                         "daily_mae": fc["daily_mae"],
                         "directional_accuracy": fc["directional_accuracy"]})
        print(f"  [{i}/{len(mine)}] {d} {t} {'ok' if fc else 'NONE'} {time.time() - t0:.0f}s",
              flush=True)
    out = config.DATA_DIR / "forecasts" / f"shard_{shard:02d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def merge() -> None:
    parts = []
    for p in sorted((config.DATA_DIR / "forecasts").glob("shard_*.csv")):
        try:
            parts.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            pass                                       # a shard with no new fits
    if FORECASTS_CSV.exists():
        parts.insert(0, pd.read_csv(FORECASTS_CSV))   # keep earlier fits
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True)
    df["date"] = df["date"].astype(str)
    df = df.drop_duplicates(["date", "ticker"], keep="first").sort_values(["date", "ticker"])
    # last_close is a raw Sharadar price; keep it out of the public results.
    df = df.drop(columns=["last_close"], errors="ignore")
    FORECASTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FORECASTS_CSV, index=False)
    print(f"[forecasts] {len(df)} rows, {int(df['ok'].sum())} ok", flush=True)


def load() -> dict[tuple[date, str], dict]:
    df = pd.read_csv(FORECASTS_CSV)
    return {(date.fromisoformat(str(r["date"])), r["ticker"]): r.to_dict()
            for _, r in df.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "shard", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan":
        j, allj = jobs(), jobs(include_done=True)
        print(f"{len(j)} new fits needed ({len(allj)} in total across variants)")
    elif a.cmd == "shard":
        run_shard(a.shard, a.nshards)
    else:
        merge()


if __name__ == "__main__":
    main()
