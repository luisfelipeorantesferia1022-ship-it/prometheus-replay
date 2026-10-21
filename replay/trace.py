"""
Lineage of every agent trade: why this ticker, why this day, why this option.

For each closed trade in each agent book it collects, from the committed
results only (no data keys needed):

  promotion week   the weekend the ticker was last promoted on or before the
                   week the trade was entered (whitelists carry over)
  stage 1          the ticker's LSTM accuracy row, gate p-value, rank, score
  stage 2          its intraday-backtest trades and the $-gate result
  sim 1 / sim 2    its trades and P&L inside each sim, plus the book totals
  promotion        the status line from that week's promotion table
  agent morning    SPY trend, LSTM forecast, scan result for the entry day
  trade            the option bought and how it ended

    python -m replay.trace            # writes results/trace/lineage.json
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from . import asof, config

R = config.RESULTS_DIR


def _csv(p) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _json(p) -> dict:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return {}


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso")) if len(df) else []


def decision_week(d: date) -> date | None:
    for a in asof.weekly_asof_dates():
        if d in asof.week_trading_days(a):
            return a
    return None


def promotion_week(variant: str, ticker: str, upto: date) -> date | None:
    tl = _csv(R / "variants" / variant / "whitelist_timeline.csv")
    if tl.empty:
        return None
    tl = tl[pd.to_datetime(tl["asof"]).dt.date <= upto]
    hits = tl[tl["promoted"].fillna("").str.split().apply(lambda l: ticker in l)]
    return date.fromisoformat(hits["asof"].iloc[-1]) if len(hits) else None


def funnel(variant: str, ticker: str, a: date) -> dict:
    wk = R / "weekly" / a.isoformat()
    vk = R / "variants" / variant / "weekly" / a.isoformat()
    acc = _csv(wk / "lstm_accuracy_results.csv")
    cand = _csv(wk / "lstm_candidates.csv")
    s1 = _json(wk / "stage1.json")
    row = acc[acc["ticker"] == ticker]
    crow = cand[cand["ticker"] == ticker]
    stage1 = {
        "swept": s1.get("swept"), "cleared_gates": s1.get("cleared_gates"),
        "candidates": s1.get("candidates", []),
        "row": _records(row)[0] if len(row) else None,
        "candidate": _records(crow)[0] if len(crow) else None,
        "dir_accuracy_rank": int((acc["dir_accuracy_pct"] > row["dir_accuracy_pct"].iloc[0]).sum() + 1)
        if len(row) else None,
    }

    bt = _csv(wk / "intraday_backtest_results.csv")
    gate = _csv(wk / "stage2_gate.csv")
    mine = bt[bt["ticker"] == ticker] if len(bt) else bt
    stage2 = {
        "trades": _records(mine[[c for c in ["date", "direction", "option_type", "strike", "expiry",
                                             "entry_premium", "contracts", "total_cost", "entry_delta",
                                             "trailing_pct", "trailing_res", "eod_pct", "oracle_pct"]
                                 if c in mine.columns]]) if len(mine) else [],
        "gate": _records(gate.sort_values("backtest_gain_pct_of_stake", ascending=False))
        if len(gate) else [],
        "threshold_pct": config.STAGE2_MIN_GAIN_PCT,
    }

    def sim(where, tag):
        tr = _csv(where / f"{tag}_closed_trades.csv")
        m = _json(where / f"{tag}_metrics.json")
        cols = ["entry_date", "exit_date", "direction", "opt_type", "strike", "entry_premium",
                "contracts", "total_cost", "pnl_pct", "pnl_dollar", "exit_reason"]
        by = (tr.groupby("ticker")["pnl_dollar"].agg(["count", "sum"]).round(2).reset_index()
              .rename(columns={"count": "trades", "sum": "pnl"}) if len(tr) else pd.DataFrame())
        return {
            "book": {k: m.get(k) for k in ["n_trades", "win_rate", "total_pnl", "return_pct",
                                          "max_dd_pct"]} if m else None,
            "by_ticker": _records(by),
            "trades": _records(tr[tr["ticker"] == ticker][[c for c in cols if c in tr.columns]])
            if len(tr) else [],
        }

    promo = _csv(vk / "promotion_table.csv")
    prow = promo[promo["ticker"] == ticker] if len(promo) else promo
    return {"asof": a.isoformat(), "stage1": stage1, "stage2": stage2,
            "sim1": sim(wk, "sim1"), "sim2": sim(vk, "sim2"),
            "promotion": _records(promo), "status": prow["status"].iloc[0] if len(prow) else None}


def build() -> dict:
    fc = _csv(R / "agent" / "lstm_forecasts.csv")
    books = []
    for summary in sorted((R / "agent").glob("*/*/summary.json")):
        bdir = summary.parent
        variant, book = bdir.parent.name, bdir.name
        trades = _csv(bdir / "trades.csv")
        log = _csv(bdir / "entry_log.csv")
        out = []
        for i, t in trades.reset_index(drop=True).iterrows():
            d = pd.Timestamp(t["entry_ts"]).date()
            wk = decision_week(d)
            pw = promotion_week(variant, t["ticker"], wk) if wk else None
            lrow = log[log["date"] == d.isoformat()] if len(log) else log
            frow = fc[(fc["date"] == d.isoformat()) & (fc["ticker"] == t["ticker"])] if len(fc) else fc
            out.append({
                "n": i + 1,
                "trade": _records(trades.iloc[[i]])[0],
                "decision_week": wk.isoformat() if wk else None,
                "promotion_week": pw.isoformat() if pw else None,
                "carried_over_weeks": (len([a for a in asof.weekly_asof_dates() if pw and pw < a <= wk])
                                       if wk and pw else None),
                "funnel": funnel(variant, t["ticker"], pw) if pw else None,
                "morning": _records(lrow)[0] if len(lrow) else None,
                "forecast": _records(frow)[0] if len(frow) else None,
            })
        books.append({"variant": variant, "book": book,
                      "summary": json.loads(summary.read_text()), "trades": out})
    return {"books": books}


def main():
    data = build()
    out = R / "trace" / "lineage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, default=str))
    n = sum(len(b["trades"]) for b in data["books"])
    print(f"[trace] {n} trades across {len(data['books'])} books -> {out}")


if __name__ == "__main__":
    main()
