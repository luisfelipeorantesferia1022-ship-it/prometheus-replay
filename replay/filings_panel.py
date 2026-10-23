"""
Monthly stock panel for the SEC filings test, built on Luis's PC from his
Sharadar bulk files (licensed: the output is never committed).

    python filings_panel.py <quant>/data/sharadar <out_dir>

Writes <out_dir>/panel_monthly.csv.gz, one row per (ticker, month-end session t)
for every ticker with a price on t, 2005-06 .. 2026-09:
  ticker, t, member (S&P 500 on t), close_unadj, ret_next (t -> next month-end,
  to the last close if it stops trading), mom12_1, rev1, mcap, bm, sector
and <out_dir>/ticker_life.csv (ticker, cik, first/last price date) used to
assign filings to tickers. Needs only pandas and numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2005-06-01", "2026-10-08"


def main(sh: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spy = pd.read_csv(sh / "derived" / "spy_daily.csv.gz", usecols=["date"], parse_dates=["date"])
    days = spy["date"].sort_values()
    days = days[(days >= START) & (days <= END)]
    me = days.groupby(days.dt.to_period("M")).max()
    me = me[me.index <= pd.Period("2026-09", "M")]           # last complete month
    month_ends = pd.DatetimeIndex(me.values)
    print("month ends", month_ends[0].date(), month_ends[-1].date(), len(month_ends), flush=True)

    parts = []
    for f in ("sp500_members_daily_1997_2008.csv.gz", "sp500_members_daily_2009_2026.csv.gz"):
        for ch in pd.read_csv(sh / "derived" / f, usecols=["ticker", "date", "close", "closeadj", "closeunadj"],
                              parse_dates=["date"], chunksize=1_000_000):
            parts.append(ch[(ch["date"] >= "2004-01-01")])
    px = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date"])
    px = px.sort_values(["ticker", "date"])
    print("daily rows", len(px), "tickers", px["ticker"].nunique(), flush=True)

    # Month-end grid: price on t exactly; next value = last close in (t, t_next].
    me_set = pd.Series(np.arange(len(month_ends)), index=month_ends)
    rows = []
    for tk, g in px.groupby("ticker", sort=False):
        g = g.set_index("date")
        ca = g["closeadj"]
        on = ca.reindex(month_ends)                           # exact month-end price or NaN
        last_by = ca.groupby(ca.index.to_period("M")).last()  # last close within each month
        last_by.index = last_by.index.to_timestamp("M")
        lb = last_by.reindex(pd.DatetimeIndex(month_ends.to_period("M").to_timestamp("M")))
        lb.index = month_ends
        nxt = lb.shift(-1)
        has_next_any = ca.index.max() > month_ends               # any price after t
        ret_next = np.where(on.notna() & nxt.notna(), nxt / on - 1.0,
                            np.where(on.notna() & ~has_next_any, 0.0, np.nan))
        # stock trades in month t+1 but has no price at all (gap) -> NaN handled above
        d = pd.DataFrame({"ticker": tk, "t": month_ends, "closeadj": on.values,
                          "close_unadj": g["closeunadj"].reindex(month_ends).values,
                          "close_split": g["close"].reindex(month_ends).values,
                          "ret_next": ret_next,
                          "mom12_1": (on.shift(1) / on.shift(12) - 1.0).values,
                          "rev1": (on / on.shift(1) - 1.0).values})
        rows.append(d[d["closeadj"].notna()])
    panel = pd.concat(rows, ignore_index=True)
    print("panel rows", len(panel), flush=True)

    mem = pd.read_csv(sh / "meta" / "sp500_membership.csv", parse_dates=["start", "end"])
    mem["end"] = mem["end"].fillna(pd.Timestamp("2099-01-01"))
    panel = panel.merge(mem, on="ticker", how="left")
    panel["member"] = (panel["t"] >= panel["start"]) & (panel["t"] < panel["end"])
    panel = (panel.sort_values("member", ascending=False)
             .drop_duplicates(["ticker", "t"]).drop(columns=["start", "end"]))

    # Fundamentals: latest quarterly (ARQ) row filed before t.
    fparts = []
    tick = set(panel["ticker"])
    for ch in pd.read_csv(sh / "bulk" / "fundamentals.csv",
                          usecols=["ticker", "dimension", "date", "sharesbas", "equity"],
                          chunksize=500_000):
        ch = ch[(ch["dimension"] == "ARQ") & ch["ticker"].isin(tick)]
        if len(ch):
            fparts.append(ch.drop(columns="dimension"))
    fu = pd.concat(fparts, ignore_index=True)
    fu["date"] = pd.to_datetime(fu["date"])
    fu = fu.dropna(subset=["date"]).sort_values("date")
    fu["fdate"] = fu["date"] + pd.Timedelta(days=1)          # filed strictly before t
    panel = panel.sort_values("t")
    panel = pd.merge_asof(panel, fu[["ticker", "fdate", "sharesbas", "equity"]].sort_values("fdate"),
                          left_on="t", right_on="fdate", by="ticker", direction="backward")
    # Sharadar SF1 share counts are restated for later splits, so pair them with
    # the split-adjusted close (not the unadjusted one).
    panel["mcap"] = panel["close_split"] * panel["sharesbas"]
    panel["bm"] = np.where(panel["equity"] > 0, panel["equity"] / panel["mcap"], np.nan)

    sec = pd.read_csv(sh / "derived" / "sp500_tickers_industry.csv", usecols=["ticker", "sector"])
    panel = panel.merge(sec.drop_duplicates("ticker"), on="ticker", how="left")
    cols = ["ticker", "t", "member", "close_unadj", "ret_next", "mom12_1", "rev1", "mcap", "bm", "sector"]
    panel = panel[cols].sort_values(["t", "ticker"])
    panel.to_csv(out / "panel_monthly.csv.gz", index=False)

    tk = pd.read_csv(sh / "bulk" / "tickers.csv",
                     usecols=["table", "ticker", "secfilings", "firstpricedate", "lastpricedate"])
    tk = tk[tk["table"] == "SF1"].drop_duplicates("ticker")
    tk["cik"] = tk["secfilings"].str.extract(r"CIK=(\d+)")[0]
    tk = tk[tk["ticker"].isin(tick) & tk["cik"].notna()]
    tk[["ticker", "cik", "firstpricedate", "lastpricedate"]].to_csv(out / "ticker_life.csv", index=False)

    m = panel[panel["member"]]
    print("members per month (min/median/max):",
          m.groupby("t").size().agg(["min", "median", "max"]).to_dict(), flush=True)
    print("coverage among members: ret_next %.3f mcap %.3f bm %.3f mom %.3f sector %.3f" % tuple(
        m[c].notna().mean() for c in ["ret_next", "mcap", "bm", "mom12_1", "sector"]), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
