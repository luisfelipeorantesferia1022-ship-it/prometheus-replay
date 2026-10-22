"""
Point-in-time views of the Sharadar subset.

Every frame handed to a strategy module ends on an as-of date and is scaled
the way the live data source would have scaled it ON THAT DATE:

  yf_adjusted(...)   = yfinance auto_adjust=True   (lstm_test)
  split_adjusted(...) = yfinance auto_adjust=False / Tradier daily history
                        (portfolio_sim_v3, intraday_backtest, lstm_forecast)

Sharadar's adjusted columns are computed with today's corporate actions. A
split or dividend after the as-of date rescales the whole earlier history by
one constant, so re-anchoring the last row to the actual unadjusted close
reproduces what the live source showed then.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from . import config

_PRICES: dict[str, pd.DataFrame] | None = None


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def prices() -> dict[str, pd.DataFrame]:
    global _PRICES
    if _PRICES is None:
        df = pd.read_parquet(config.SHARADAR_DIR / "prices.parquet")
        df["date"] = pd.to_datetime(df["date"])
        _PRICES = {t: g.set_index("date").sort_index().drop(columns="ticker")
                   for t, g in df.groupby("ticker")}
    return _PRICES


def set_prices(frames: dict[str, pd.DataFrame]) -> None:
    """Inject frames directly (tests)."""
    global _PRICES
    _PRICES = frames


@lru_cache(maxsize=1)
def sp500_table() -> pd.DataFrame:
    df = pd.read_parquet(config.SHARADAR_DIR / "sp500.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@lru_cache(maxsize=1)
def _earnings() -> dict[str, list[date]]:
    p = config.SHARADAR_DIR / "earnings.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out: dict[str, list[date]] = {}
    for t, g in df.groupby("ticker"):
        out[t] = sorted(pd.to_datetime(g["date"]).dt.date.unique())
    return out


def to_sharadar(ticker: str) -> str:
    """Live code uses yfinance style BRK-B; Sharadar uses BRK.B."""
    return ticker.replace("-", ".")


def to_live(ticker: str) -> str:
    return ticker.replace(".", "-")


# ─────────────────────────────────────────────────────────────────────────────
#  Membership
# ─────────────────────────────────────────────────────────────────────────────

def sp500_members(sp: pd.DataFrame, asof: date) -> set[str]:
    """
    Members as of `asof`: the latest quarterly 'historical' snapshot on or
    before the date, then every 'added' / 'removed' row after the snapshot up
    to and including the date.
    """
    asof = pd.Timestamp(asof)
    snaps = sp[(sp["action"] == "historical") & (sp["date"] <= asof)]
    if snaps.empty:
        raise ValueError(f"no S&P 500 snapshot on or before {asof.date()}")
    snap_date = snaps["date"].max()
    members = set(snaps.loc[snaps["date"] == snap_date, "ticker"])
    changes = sp[(sp["date"] > snap_date) & (sp["date"] <= asof)
                 & sp["action"].isin(["added", "removed"])].sort_values("date")
    for _, r in changes.iterrows():
        if r["action"] == "added":
            members.add(r["ticker"])
        else:
            members.discard(r["ticker"])
    return members


def members_live_format(asof: date) -> list[str]:
    """What lstm_test.get_sp500_tickers() would have returned that weekend."""
    return sorted(to_live(t) for t in sp500_members(sp500_table(), asof))


# ─────────────────────────────────────────────────────────────────────────────
#  Price frames
# ─────────────────────────────────────────────────────────────────────────────

def _raw(ticker: str) -> pd.DataFrame | None:
    return prices().get(to_sharadar(ticker))


def _window(ticker: str, start, end_exclusive) -> pd.DataFrame | None:
    df = _raw(ticker)
    if df is None:
        return None
    w = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end_exclusive))]
    return w if not w.empty else None


def yf_adjusted(ticker: str, start, end_exclusive) -> pd.DataFrame:
    """yf.download(..., auto_adjust=True) as it looked on the last row's date."""
    w = _window(ticker, start, end_exclusive)
    if w is None:
        return pd.DataFrame()
    last = w.iloc[-1]
    anchor = float(last["closeunadj"]) / float(last["closeadj"])
    adj_close = w["closeadj"] * anchor
    ratio = adj_close / w["close"]
    return pd.DataFrame({
        "Open": w["open"] * ratio,
        "High": w["high"] * ratio,
        "Low": w["low"] * ratio,
        "Close": adj_close,
        "Volume": w["volume"] * (float(last["close"]) / float(last["closeunadj"])),
    })


def split_adjusted(ticker: str, start, end_exclusive) -> pd.DataFrame:
    """Split-adjusted OHLCV as of the last row's date (no dividend adjustment)."""
    w = _window(ticker, start, end_exclusive)
    if w is None:
        return pd.DataFrame()
    last = w.iloc[-1]
    s = float(last["closeunadj"]) / float(last["close"])
    return pd.DataFrame({
        "Open": w["open"] * s,
        "High": w["high"] * s,
        "Low": w["low"] * s,
        "Close": w["close"] * s,
        "Volume": w["volume"] / s,
    })


def unadjusted_close(ticker: str, d: date) -> float | None:
    df = _raw(ticker)
    if df is None:
        return None
    w = df[df.index <= pd.Timestamp(d)]
    return float(w["closeunadj"].iloc[-1]) if not w.empty else None


def earnings_dates(ticker: str) -> list[date]:
    """
    Earnings-release dates (8-K Item 2.02 filing dates). The live code reads
    yfinance's scheduled + historical dates; a scheduled date is public weeks
    ahead, so using the actual filing date as of an earlier day is fair.
    """
    return list(_earnings().get(to_sharadar(ticker), []))


# ─────────────────────────────────────────────────────────────────────────────
#  Calendar
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def nyse_schedule() -> pd.DataFrame:
    import pandas_market_calendars as mcal
    s = mcal.get_calendar("NYSE").schedule(config.CALENDAR_START,
                                           config.CALENDAR_END or config.AGENT_LAST_EXIT_DAY)
    s["open_et"] = s["market_open"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    s["close_et"] = s["market_close"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    s.index = s.index.date
    return s


def trading_days(start, end) -> list[date]:
    s = nyse_schedule()
    a, b = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return [d for d in s.index if a <= d <= b]


def prev_trading_day(d: date) -> date:
    days = [x for x in nyse_schedule().index if x < d]
    return days[-1]


def weekly_asof_dates() -> list[date]:
    """Last trading day of each week, first..last as-of in config."""
    days = trading_days(config.REPLAY_FIRST_ASOF, config.REPLAY_LAST_ASOF)
    by_week: dict[tuple, date] = {}
    for d in days:
        by_week[d.isocalendar()[:2]] = d
    return sorted(by_week.values())


def week_trading_days(asof: date) -> list[date]:
    """Trading days governed by the whitelist decided after `asof`."""
    all_asof = weekly_asof_dates()
    i = all_asof.index(asof)
    start = asof + timedelta(days=1)
    end = (all_asof[i + 1] if i + 1 < len(all_asof)
           else pd.Timestamp(config.AGENT_LAST_ENTRY_DAY).date())
    return trading_days(start, end)
