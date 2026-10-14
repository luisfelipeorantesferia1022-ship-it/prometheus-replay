"""
Sharadar bulk download -> small parquet subset used by every replay stage.

Only what the replay needs is kept on disk:
  prices.parquet    daily OHLCV for every ticker that was in the S&P 500 at any
                    point in MEMBERSHIP_WINDOW, plus SPY (from `funds`)
  sp500.parquet     the full membership table (snapshots + adds/removes)
  earnings.parquet  8-K Item 2.02 filing dates (eventcode 22) per ticker

Raw zips are deleted after filtering. Nothing here is ever committed: the data
directory is git-ignored and only lives in the Actions cache.

    python -m replay.sharadar            # needs SHARADAR_API_KEY
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

API = "https://api.sharadar.com/v1.0/data/{table}"
EARNINGS_EVENTCODE = "22"   # Item 2.02 Results of Operations (verified on AAPL)

PRICE_COLS = ["ticker", "date", "open", "high", "low", "close", "volume",
              "closeadj", "closeunadj"]


def _key() -> str:
    k = os.environ.get("SHARADAR_API_KEY", "").strip()
    if not k:
        sys.exit("SHARADAR_API_KEY is not set")
    return k


def download_bulk(table: str, years: str, dest: Path) -> Path:
    """GET /data/{table}?years=... -> 302 to a pre-signed zip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = API.format(table=table)
    print(f"[sharadar] bulk {table} years={years} ...", flush=True)
    try:
        r = requests.get(url, params={"api_key": _key(), "years": years},
                         stream=True, timeout=600, allow_redirects=True)
    except requests.RequestException as e:
        # Exception text can contain the request URL, which carries the key.
        raise RuntimeError(f"[sharadar] {table}: {type(e).__name__}") from None
    with r:
        if r.status_code != 200:
            # Never echo the URL: it carries the key.
            raise RuntimeError(f"[sharadar] {table}: HTTP {r.status_code}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    print(f"[sharadar] {table}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)
    return dest


def _read_zip_csv(path: Path, usecols=None, chunksize=None):
    zf = zipfile.ZipFile(path)
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError(f"no csv inside {path.name}: {zf.namelist()}")
    fh = zf.open(names[0])
    return pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"),
                       usecols=usecols, chunksize=chunksize, low_memory=False)


def members_in_window(sp500: pd.DataFrame, start: str, end: str) -> set[str]:
    from .asof import sp500_members
    out: set[str] = set()
    for d in pd.date_range(start, end, freq="W-FRI"):
        out |= sp500_members(sp500, d.date())
    return out


def build(keep_raw: bool = False) -> None:
    out = config.SHARADAR_DIR
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)

    # ── Membership first: it decides which price rows to keep ──────────────
    p = download_bulk("sp500", "full", raw / "sp500.zip")
    sp = _read_zip_csv(p)
    sp["date"] = pd.to_datetime(sp["date"])
    sp.to_parquet(out / "sp500.parquet", index=False)
    universe = members_in_window(sp, *config.MEMBERSHIP_WINDOW)
    print(f"[sharadar] S&P 500 members in window: {len(universe)}", flush=True)

    start = pd.Timestamp(config.DATA_START)
    end = pd.Timestamp(config.DATA_END)

    def _filter(table: str, years: str, tickers: set[str]) -> pd.DataFrame:
        p = download_bulk(table, years, raw / f"{table}.zip")
        parts = []
        for ch in _read_zip_csv(p, chunksize=2_000_000):
            ch = ch[ch["ticker"].isin(tickers)]
            if ch.empty:
                continue
            ch = ch.assign(date=pd.to_datetime(ch["date"]))
            ch = ch[(ch["date"] >= start) & (ch["date"] <= end)]
            parts.append(ch[[c for c in PRICE_COLS if c in ch.columns]])
        if not keep_raw:
            p.unlink(missing_ok=True)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PRICE_COLS)

    stocks = _filter("stocks", "10", universe)
    funds = _filter("funds", "10", {"SPY"})
    prices = pd.concat([stocks, funds], ignore_index=True)
    prices = prices.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    prices.to_parquet(out / "prices.parquet", index=False)
    have = set(prices["ticker"].unique())
    missing = sorted(universe - have)
    print(f"[sharadar] prices: {len(prices):,} rows, {len(have)} tickers; "
          f"{len(missing)} members without price rows", flush=True)
    (out / "missing_price_tickers.txt").write_text("\n".join(missing))

    # ── Earnings dates ─────────────────────────────────────────────────────
    p = download_bulk("events", "10", raw / "events.zip")
    ev = _read_zip_csv(p)
    ev["eventcodes"] = ev["eventcodes"].astype(str)
    ev = ev[ev["eventcodes"].str.split("|").apply(lambda c: EARNINGS_EVENTCODE in c)]
    ev = ev[ev["ticker"].isin(universe)][["ticker", "date"]]
    ev["date"] = pd.to_datetime(ev["date"])
    ev.to_parquet(out / "earnings.parquet", index=False)
    print(f"[sharadar] earnings filings: {len(ev):,}", flush=True)
    if not keep_raw:
        p.unlink(missing_ok=True)
        (raw / "sp500.zip").unlink(missing_ok=True)


if __name__ == "__main__":
    build(keep_raw="--keep-raw" in sys.argv)
