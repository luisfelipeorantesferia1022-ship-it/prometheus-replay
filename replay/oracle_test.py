"""
Oracle test (2026-10-23): which target should a model predict for V3?

For every stock-week where both the call and the put were tradable and measured
with real quotes (2024 and Jan-Sep 2026 from results/oos/oos_trades.csv, 2025
from the calm test), ask: if a model knew X perfectly and bought the call when X
said up and the put when it said down, what would V3 have earned, and what
accuracy would it need just to break even?  X is the stock's direction from the
9:46 entry to: the entry day's close (what the LSTM forecasts), the next day's
close, day 2, day 5, the Thursday exit, or whichever way it first moves 3%.

Run on Luis's PC against the Sharadar bulk stocks.csv (date-filtered with
  awk -F, 'NR==1 || ($2 >= "2023-12-01" && $2 <= "2026-10-31")' stocks.csv > sep_sub.csv)
next to oos_trades.csv, calm_test.csv and exit_v3_check.csv:
    python oracle_test.py <folder>       # output: results/analysis/oracle_test.txt
"""
import sys

import numpy as np
import pandas as pd

H = sys.argv[1] if len(sys.argv) > 1 else "."
X = 3.0   # first-passage threshold, % stock move (about what takes a V3 option to +50%)


def load():
    oos = pd.read_csv(f"{H}/oos_trades.csv")
    oos["year"] = oos["period"].astype(str)
    oos["tradable"] = oos["tradable"].astype("boolean").fillna(False).astype(bool)
    calm = pd.read_csv(f"{H}/calm_test.csv")
    v3 = pd.read_csv(f"{H}/exit_v3_check.csv").set_index("key")
    calm["v3_pct"] = (calm["raw_symbol"] + "|" + calm["entry_date"]).map(v3["v3_pct"])
    calm["year"] = "2025"
    calm["tradable"] = (calm["spread_pct"] <= 20) & (calm["delta"].abs() >= 0.35)
    cols = ["year", "asof", "entry_date", "ticker", "side", "stock_at_entry", "end_day", "tradable", "v3_pct"]
    t = pd.concat([oos[cols], calm[cols]], ignore_index=True)
    t = t[t["tradable"] & t["v3_pct"].notna() & t["stock_at_entry"].notna()].copy()
    t["sym"] = t["ticker"].str.replace("-", ".", regex=False)
    sep = pd.read_csv(f"{H}/sep_sub.csv", usecols=["ticker", "date", "high", "low", "close", "closeunadj"])
    sep = sep[sep["ticker"].isin(set(t["sym"]))].sort_values(["ticker", "date"])
    return t, {k: g.set_index("date") for k, g in sep.groupby("ticker")}


def path(px, sym, d, e, p0):
    g = px.get(sym)
    if g is None:
        return None
    w = g[g.index >= d]
    if not len(w) or w.index[0] != d:
        return None
    base = p0 / (w["closeunadj"].iloc[0] / w["close"].iloc[0])   # 9:46 price on the split-adjusted scale
    c, hi, lo = ((w[k] / base - 1) * 100 for k in ("close", "high", "low"))
    out = {"r0": c.iloc[0], **{f"r{k}": c.iloc[k] if len(c) > k else np.nan for k in (1, 2, 5)},
           "rh": c.loc[e] if e in c.index else np.nan}
    fp = 1.0 if c.iloc[0] >= X else -1.0 if c.iloc[0] <= -X else 0.0   # day 0: close only
    for k in range(1, len(c)):
        if fp or c.index[k] > e:
            break
        up, dn = hi.iloc[k] >= X, lo.iloc[k] <= -X
        fp = np.sign(c.iloc[k]) if up and dn else 1.0 if up else -1.0 if dn else 0.0
    out["fp"] = fp if fp else np.sign(out["rh"])
    return out


def main():
    t, px = load()
    u = t.drop_duplicates(["year", "asof", "ticker"])
    m = pd.DataFrame([{"year": r.year, "asof": r.asof, "ticker": r.ticker, **p} for r in u.itertuples()
                      if (p := path(px, r.sym, r.entry_date, r.end_day, r.stock_at_entry))])
    w = t.pivot_table(index=["year", "asof", "ticker"], columns="side", values="v3_pct", aggfunc="first").dropna()
    targets = {"rest of entry day (what the LSTM forecasts)": "r0", "through next day's close": "r1",
               "through day 2": "r2", "through day 5": "r5", "whole hold (to the Thursday exit)": "rh",
               f"first to move {X:.0f}% either way": "fp"}
    w = w.reset_index().merge(m, on=["year", "asof", "ticker"]).dropna(subset=list(targets.values()))
    print(f"pairs (both sides tradable, same stock-week): {len(w)} {w['year'].value_counts().sort_index().to_dict()}")
    for yr in ["2024", "2025", "2026", "all"]:
        x = w if yr == "all" else w[w["year"] == yr]
        print(f"\n=== {yr}: {len(x)} pairs | coin-flip side {pd.concat([x['CALL'], x['PUT']]).mean():+.1f}% | "
              f"always call {x['CALL'].mean():+.1f}% | stock up over 5 days {(x['r5'] > 0).mean() * 100:.0f}% ===")
        best = np.where(x["CALL"] >= x["PUT"], 1, -1)
        A, B = x[["CALL", "PUT"]].max(axis=1).mean(), x[["CALL", "PUT"]].min(axis=1).mean()
        print(f"  {'best side in hindsight (upper bound)':46s} right {A:+6.1f}% wrong {B:+6.1f}% break-even {-B / (A - B) * 100:3.0f}%")
        for name, col in targets.items():
            s = np.sign(x[col]).replace(0, 1)
            A = np.where(s > 0, x["CALL"], x["PUT"]).mean()
            B = np.where(s > 0, x["PUT"], x["CALL"]).mean()
            at = " / ".join(f"{p * A + (1 - p) * B:+.1f}%" for p in (0.55, 0.60, 0.65))
            print(f"  {name:46s} right {A:+6.1f}% wrong {B:+6.1f}% break-even {-B / (A - B) * 100:3.0f}% | "
                  f"agrees with best side {(s.values == best).mean() * 100:.0f}% | at 55/60/65% accuracy {at}")
    t2 = t.merge(m, on=["year", "asof", "ticker"])
    sgn = np.where(t2["side"] == "CALL", 1, -1)
    print("\nRank correlation of each option's V3 return with the stock's move its way:")
    for name, col in targets.items():
        if col != "fp":
            ok = t2[col].notna()
            print(f"  {name:46s} {(t2.loc[ok, col] * sgn[ok]).rank().corr(t2.loc[ok, 'v3_pct'].rank()):+.2f}")


if __name__ == "__main__":
    main()
