"""
Independent re-implementation of the filings-test portfolios (pandas, row by
row), written separately from replay/filings_eval.py. It reads the same inputs
and must reproduce every month's gross and net return of every setting.

    python tests/filings_verify.py --panel <dir> --signals results/filings/signals_monthly.csv.gz \
        --compare results/filings/search_monthly_net.csv --months 2007-02:2018-12
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def build_books(panel: pd.DataFrame, sig: str) -> dict:
    """Formation book per month-end t: {ticker: weight}."""
    books = {}
    for t, g in panel.groupby("t"):
        g = g[g["member"] & (g["close_unadj"] >= 5) & g["ret_next"].notna()
              & g[sig].notna() & g["sector"].notna()]
        if len(g) < 50:
            books[t] = {}
            continue
        size = g.groupby("sector")["ticker"].transform("size")
        grp = np.where(size < 10, "zz_other", g["sector"])
        g = g.assign(grp=grp)
        g["pct"] = g.groupby("grp")[sig].transform(
            lambda v: (v.rank(method="average") - 0.5) / len(v))
        lg, sh = g[g["pct"] > 0.8], g[g["pct"] < 0.2]
        if lg.empty or sh.empty:
            books[t] = {}
            continue
        b = {tk: 1 / len(lg) for tk in lg["ticker"]}
        b.update({tk: -1 / len(sh) for tk in sh["ticker"]})
        books[t] = b
    return books


def run(panel: pd.DataFrame, sig: str, K: int, a: str, b: str) -> pd.DataFrame:
    books = build_books(panel, sig)
    ts = sorted(books)
    ret = panel.set_index(["t", "ticker"])["ret_next"]
    out, prev = [], {}
    for i, t in enumerate(ts):
        held: dict = {}
        for j in range(K):
            if i - j < 0:
                continue
            for tk, w in books[ts[i - j]].items():
                held[tk] = held.get(tk, 0.0) + w / K
        keys = set(held) | set(prev)
        turn = sum(abs(held.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        gross = sum(w * (0.0 if pd.isna(ret.get((t, tk), np.nan)) else ret.get((t, tk)))
                    for tk, w in held.items())
        short = sum(-w for w in held.values() if w < 0)
        net = gross - 0.00075 * turn - 0.005 / 12 * short
        month = str((pd.Period(t, "M") + 1))
        out.append({"month": month, "gross": gross, "net": net})
        prev = held
    df = pd.DataFrame(out)
    return df[(df["month"] >= a) & (df["month"] <= b)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--months", required=True)
    x = ap.parse_args()
    a, b = x.months.split(":")
    p = pd.read_csv(f"{x.panel}/panel_monthly.csv.gz", parse_dates=["t"])
    s = pd.read_csv(x.signals, parse_dates=["t"])
    p = p.merge(s, on=["ticker", "t"], how="left")
    lo = pd.Period(a, "M") - 5
    p = p[p["t"].dt.to_period("M") >= lo]
    ref = pd.read_csv(x.compare, index_col=0)
    worst = 0.0
    for col in ref.columns:
        sig, K = col.split("_K")
        mine = run(p, sig, int(K), a, b).set_index("month")["net"]
        d = (mine - ref[col].reindex(mine.index)).abs().max()
        worst = max(worst, d)
        print(f"{col}: {len(mine)} months, max |difference| {d:.2e}")
    print("MATCH" if worst < 1e-9 else "MISMATCH", worst)


if __name__ == "__main__":
    main()
