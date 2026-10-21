"""
Databento access with a cost quote before every download, a hard budget, and
a cache so nothing is ever bought twice.

Spending is recorded in a ledger kept in two places: the data cache and
results/databento_ledger.json (committed, so the budget survives a cache
eviction). The larger of the two totals is treated as already spent.

Set DATABENTO_QUOTE_ONLY=1 to price requests without downloading anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config

ET = "America/New_York"
QUOTE_ONLY = os.environ.get("DATABENTO_QUOTE_ONLY", "") == "1"


class BudgetExceeded(RuntimeError):
    pass


_client = None


def client():
    global _client
    if _client is None:
        import databento as db
        key = os.environ.get("DATABENTO_API_KEY", "").strip()
        if not key:
            raise RuntimeError("DATABENTO_API_KEY is not set")
        _client = db.Historical(key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
#  Ledger
# ─────────────────────────────────────────────────────────────────────────────

_LEDGER_CACHE = config.DBN_CACHE_DIR / "ledger.json"
_LEDGER_REPO = config.RESULTS_DIR / "databento_ledger.json"


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"spent_usd": 0.0, "quoted_usd": 0.0, "requests": 0, "by_purpose": {}}


def ledger() -> dict:
    a, b = _read(_LEDGER_CACHE), _read(_LEDGER_REPO)
    return a if a.get("spent_usd", 0) >= b.get("spent_usd", 0) else b


def _write(led: dict) -> None:
    for p in (_LEDGER_CACHE, _LEDGER_REPO):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(led, indent=2, sort_keys=True))


def remaining() -> float:
    return config.DATABENTO_BUDGET_USD - ledger()["spent_usd"]


def _charge(cost: float, purpose: str, quoted_only: bool) -> None:
    led = ledger()
    key = "quoted_usd" if quoted_only else "spent_usd"
    led[key] = round(led.get(key, 0.0) + cost, 6)
    if not quoted_only:
        led["requests"] = led.get("requests", 0) + 1
        bp = led.setdefault("by_purpose", {})
        bp[purpose] = round(bp.get(purpose, 0.0) + cost, 6)
    _write(led)


RETRY_DELAYS = [5, 15, 30, 60, 120, 180, 300, 300]   # ~17 minutes in total


def _transient(e: Exception) -> bool:
    """Server-side or network trouble worth waiting out (5xx, 429, timeouts)."""
    status = getattr(e, "http_status", None)
    if status is not None:
        return status >= 500 or status == 429
    name = type(e).__name__
    return any(k in name for k in ("Timeout", "Connection", "RemoteDisconnected",
                                   "ChunkedEncoding", "ProtocolError"))


def _with_retry(fn, what: str):
    for i, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            return fn()
        except Exception as e:
            if not _transient(e) or i == len(RETRY_DELAYS):
                raise
            print(f"[dbn] {what}: {type(e).__name__} "
                  f"{getattr(e, 'http_status', '')} - retrying in {RETRY_DELAYS[i]}s", flush=True)


def _get(purpose: str, cache: Path, **params) -> pd.DataFrame | None:
    """Quote, check budget, download, cache. Returns None in quote-only mode."""
    if cache.exists():
        return pd.read_parquet(cache)
    c = client()
    cost = float(_with_retry(lambda: c.metadata.get_cost(**params), f"{purpose} quote"))
    if QUOTE_ONLY:
        _charge(cost, purpose, quoted_only=True)
        return None
    if cost > remaining():
        raise BudgetExceeded(
            f"{purpose}: request costs ${cost:.4f}, ${remaining():.4f} left of "
            f"${config.DATABENTO_BUDGET_USD:.2f}")
    data = _with_retry(lambda: c.timeseries.get_range(**params), f"{purpose} download")
    df = data.to_df()
    _charge(cost, purpose, quoted_only=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Equity minute bars
# ─────────────────────────────────────────────────────────────────────────────

def equity_dataset() -> str:
    led = ledger()
    if led.get("equity_dataset"):
        return led["equity_dataset"]
    need = pd.Timestamp(config.MEMBERSHIP_WINDOW[0]) - pd.Timedelta(days=90)
    chosen = None
    for ds in config.EQUITY_DATASETS:
        try:
            rng = client().metadata.get_dataset_range(ds)
            start = pd.Timestamp(rng["start"] if isinstance(rng, dict) else rng.start)
            if start.tz_localize(None) <= need:
                chosen = ds
                break
            print(f"[dbn] {ds} starts {start.date()}, too late", flush=True)
        except Exception as e:
            print(f"[dbn] {ds} unavailable: {e}", flush=True)
    if not chosen:
        raise RuntimeError("no Databento equity dataset covers the window")
    led["equity_dataset"] = chosen
    _write(led)
    print(f"[dbn] equity dataset: {chosen}", flush=True)
    return chosen


def _db_sym(ticker: str) -> str:
    return ticker.replace("-", ".")


def _day_window_utc(d: date, start_hm="09:00", end_hm="16:01"):
    s = pd.Timestamp(f"{d} {start_hm}", tz=ET).tz_convert("UTC")
    e = pd.Timestamp(f"{d} {end_hm}", tz=ET).tz_convert("UTC")
    return s, e


def _eq_cache(ticker: str, d: date) -> Path:
    return config.DBN_CACHE_DIR / "eq1m" / d.isoformat() / f"{_db_sym(ticker)}.parquet"


def prefetch_equity_day(d: date, tickers: list[str], purpose: str) -> None:
    """One request per day for every ticker still missing that day."""
    missing = sorted({t for t in tickers if not _eq_cache(t, d).exists()})
    if not missing:
        return
    s, e = _day_window_utc(d)
    cache = config.DBN_CACHE_DIR / "eq1m_batch" / f"{d}_{_hash(missing)}.parquet"
    df = _get(purpose, cache, dataset=equity_dataset(), schema="ohlcv-1m",
              symbols=[_db_sym(t) for t in missing], stype_in="raw_symbol",
              start=s, end=e)
    if df is None:   # quote-only
        return
    df = df.reset_index()
    for t in missing:
        sub = df[df["symbol"] == _db_sym(t)] if "symbol" in df.columns else df.iloc[0:0]
        p = _eq_cache(t, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(p)
    cache.unlink(missing_ok=True)


def equity_bars(ticker: str, d: date, purpose: str = "equity") -> pd.DataFrame:
    """
    1-minute bars for one day, 09:00-16:00 ET, indexed by bar START time in
    naive ET. Columns open/high/low/close/volume. Empty frame if none.
    """
    p = _eq_cache(ticker, d)
    if not p.exists():
        prefetch_equity_day(d, [ticker], purpose)
    if not p.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.read_parquet(p)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["ts_event"], utc=True) if "ts_event" in df.columns else df.index
    idx = pd.DatetimeIndex(ts).tz_convert(ET).tz_localize(None)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index = idx
    return out.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
#  Options (OPRA)
# ─────────────────────────────────────────────────────────────────────────────

def _hash(items) -> str:
    return hashlib.sha1("|".join(map(str, items)).encode()).hexdigest()[:16]


def option_definitions(underlying: str, d: date) -> pd.DataFrame:
    """All listed contracts on `underlying` as published at the start of day d."""
    cache = config.DBN_CACHE_DIR / "opra_def" / d.isoformat() / f"{_db_sym(underlying)}.parquet"
    s = pd.Timestamp(f"{d} 00:00", tz="UTC")
    df = _get("options_definitions", cache, dataset=config.OPTIONS_DATASET,
              schema="definition", symbols=[f"{_db_sym(underlying)}.OPT"],
              stype_in="parent", start=s, end=s + pd.Timedelta(hours=14))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df = df.sort_values("ts_recv").drop_duplicates("raw_symbol", keep="last")
    # Expiry date straight from the OSI symbol (root padded to 6, then YYMMDD).
    # Converting the UTC expiration timestamp to New York time lands on the
    # previous day, which moved every expiry (and the Thursday force-exit)
    # one day early.
    df["expiration"] = pd.to_datetime(df["raw_symbol"].str[6:12], format="%y%m%d").dt.date
    return df


def option_quotes(raw_symbols: list[str], start_et: pd.Timestamp, end_et: pd.Timestamp,
                  purpose: str) -> pd.DataFrame:
    """CBBO at 1-minute intervals (bid/ask/last sale), naive-ET index."""
    syms = sorted(set(raw_symbols))
    cache = config.DBN_CACHE_DIR / "opra_cbbo1m" / f"{_hash(syms + [start_et, end_et])}.parquet"
    df = _get(purpose, cache, dataset=config.OPTIONS_DATASET, schema="cbbo-1m",
              symbols=syms, stype_in="raw_symbol",
              start=pd.Timestamp(start_et, tz=ET).tz_convert("UTC"),
              end=pd.Timestamp(end_et, tz=ET).tz_convert("UTC"))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    ts = pd.to_datetime(df["ts_recv"] if "ts_recv" in df.columns else df["ts_event"], utc=True)
    df["ts_et"] = ts.dt.tz_convert(ET).dt.tz_localize(None)
    df = df.rename(columns={"bid_px_00": "bid", "ask_px_00": "ask"})
    return df.sort_values("ts_et")


def option_trades(raw_symbol: str, start_et: pd.Timestamp, end_et: pd.Timestamp,
                  purpose: str) -> pd.DataFrame:
    cache = config.DBN_CACHE_DIR / "opra_trades" / f"{_hash([raw_symbol, start_et, end_et])}.parquet"
    df = _get(purpose, cache, dataset=config.OPTIONS_DATASET, schema="trades",
              symbols=[raw_symbol], stype_in="raw_symbol",
              start=pd.Timestamp(start_et, tz=ET).tz_convert("UTC"),
              end=pd.Timestamp(end_et, tz=ET).tz_convert("UTC"))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    ts = pd.to_datetime(df["ts_event"] if "ts_event" in df.columns else df["ts_recv"], utc=True)
    df["ts_et"] = ts.dt.tz_convert(ET).dt.tz_localize(None)
    return df.sort_values("ts_et")
