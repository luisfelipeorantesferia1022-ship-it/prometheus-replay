"""
Independent re-computation of the pairs-search holdout for the selected setting,
written without replay/pairs_search.py (only the shared index-membership helper
is reused). Re-derives the universe, the S1 distance pairs, the 60-day z-scores
(pandas rolling), every trade and the daily portfolio return, and compares them
with the module's outputs.

    python tests/pairs_search_verify.py DATA_DIR SCRATCH_DIR
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay.asof import sp500_members  # noqa: E402

DATA, SCR = Path(sys.argv[1]), Path(sys.argv[2])
RES = Path(__file__).resolve().parent.parent / "results" / "pairs_search"
sel = json.loads((RES / "selection.json").read_text())["selected"]
assert (sel["method"], str(sel["lookback"]), sel["entry_z"], sel["exit_within"], sel["stop"], sel["max_hold"]) == \
    ("S1_distance", "60", 1.0, 0.0, "none", 60), sel
L, ENTRY, HOLD, COST, BORROW = 60, 1.0, 60, 0.0015, 0.005 * 0.5 / 252

df = pd.concat([pd.read_csv(DATA / f, parse_dates=["date"]) for f in
                ("sp500_members_daily_1997_2008.csv.gz", "sp500_members_daily_2009_2026.csv.gz")])
spy = pd.read_csv(DATA / "spy_daily.csv.gz", parse_dates=["date"]).set_index("date")
cal = spy.index[(spy.index >= df["date"].min()) & (spy.index <= df["date"].max())].sort_values()
wide = {c: df.pivot(index="date", columns="ticker", values=c).reindex(cal) for c in
        ("open", "close", "closeadj", "closeunadj", "volume")}
CA = wide["closeadj"]
OA = wide["open"] * wide["closeadj"] / wide["close"]
DV = wide["close"] * wide["volume"]
sp = pd.read_csv(DATA / "sp500_membership_raw.csv", parse_dates=["date"])
tk = pd.read_csv(DATA / "sp500_tickers_industry.csv").set_index("ticker")
industry = tk["industry"].where(tk["industry"].notna(), tk["sicindustry"])


def company(s):
    s = re.sub(r"[^A-Z0-9]+", " ", str(s).upper())
    s = re.sub(r"\b(?:CLASS|CL|SERIES|SER) [A-Z]\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


names = tk["name"].map(company)

# formation dates: first session on/after Jan 1 and Jul 1
starts = [cal[cal >= pd.Timestamp(f"{y}-{m:02d}-01")][0] for y in range(1999, 2027) for m in (1, 7)
          if (cal >= pd.Timestamp(f"{y}-{m:02d}-01")).any()]
hold_starts = [d for d in starts if d.year >= 2013]
pos_of = {d: i for i, d in enumerate(cal)}

trades = []
daily = pd.Series(0.0, index=cal)
for fi, F in enumerate(hold_starts):
    k = pos_of[F]
    nxt = starts[starts.index(F) + 1] if starts.index(F) + 1 < len(starts) else None
    k2 = pos_of[nxt] if nxt is not None else len(cal)
    prev = cal[k - 1]
    mem = [t for t in sp500_members(sp, prev.date()) if t in CA.columns]
    win = CA.iloc[k - 252:k][mem]
    mem = [t for t in mem if win[t].notna().all() and wide["closeunadj"].iloc[k - 1][t] >= 5]
    dv = DV.iloc[k - 60:k][mem].mean()
    top = (pd.DataFrame({"t": dv.index, "dv": dv.values}).fillna({"dv": -1.0})
           .sort_values(["dv", "t"], ascending=[False, True]).head(300)["t"].tolist())
    norm = CA.iloc[k - 252:k][top]
    norm = norm / norm.iloc[0]
    cand = []
    by_ind = pd.Series(top).groupby(pd.Series(top).map(industry)).apply(list)
    for g, members in by_ind.items():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = sorted((members[x], members[y]))
                if names.get(a) and names.get(a) == names.get(b):
                    continue
                cand.append((float(((norm[a] - norm[b]) ** 2).sum()), a, b))
    used, pairs = set(), []
    for ssd, a, b in sorted(cand):
        if a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
        if len(pairs) == 20:
            break
    for a, b in pairs:
        ca_all = CA.iloc[k - 252:k2][[a, b]]
        both = ca_all.iloc[252:].notna().all(axis=1).to_numpy()
        if not both.any():
            continue
        n = int(np.flatnonzero(both)[-1]) + 1
        ca_all = ca_all.iloc[:252 + n].ffill()
        spread = np.log(ca_all[a]) - np.log(ca_all[b])
        z = ((spread - spread.rolling(L).mean()) / spread.rolling(L).std()).iloc[252:].to_numpy()
        cA, cB = ca_all[a].iloc[252:].to_numpy(), ca_all[b].iloc[252:].to_numpy()
        oA = OA[a].iloc[k:k + n].to_numpy()
        oB = OA[b].iloc[k:k + n].to_numpy()
        prevA = ca_all[a].iloc[251:251 + n].to_numpy()
        prevB = ca_all[b].iloc[251:251 + n].to_numpy()
        oA = np.where(np.isnan(oA), prevA, oA)
        oB = np.where(np.isnan(oB), prevB, oB)
        dates = cal[k:k + n]
        data_end = n < (k2 - k)
        t, armed = 0, True
        while t < n:
            zt = z[t]
            if not np.isnan(zt):
                if not armed and abs(zt) < ENTRY / 2:
                    armed = True
                if armed and abs(zt) >= ENTRY and t + 1 <= n - 2:
                    side = 1 if zt < 0 else -1          # +1: long a, short b
                    e, armed = t + 1, False
                    d = e
                    while True:
                        if d == n - 1:
                            x, at_open, why = d, False, "data end" if data_end else "period end"
                            break
                        if not np.isnan(z[d]) and side * z[d] >= 0:
                            x, at_open, why = d + 1, True, "converged"
                            break
                        if d - e + 1 >= HOLD:
                            x, at_open, why = d + 1, True, "time"
                            break
                        d += 1
                    wa, wb = 0.5 * side, -0.5 * side
                    pa, pb = oA[e], oB[e]
                    last = x - 1 if at_open else x
                    net = 0.0
                    for j in range(e, last + 1):
                        v = wa * (cA[j] - pa) / oA[e] + wb * (cB[j] - pb) / oB[e] - BORROW
                        daily[dates[j]] += v / 20
                        net += v
                        pa, pb = cA[j], cB[j]
                    if at_open:
                        v = wa * (oA[x] - pa) / oA[e] + wb * (oB[x] - pb) / oB[e]
                        daily[dates[x]] += v / 20
                        net += v
                    daily[dates[e]] -= COST / 2 / 20
                    daily[dates[x]] -= COST / 2 / 20
                    net -= COST
                    trades.append({"formation": str(F.date()), "pair": f"{a}/{b}", "entry": dates[e], "exit": dates[x],
                                   "reason": why, "net": net})
                    if not at_open:
                        break
                    t = x
                    continue
            t += 1

mine = pd.DataFrame(trades)
hd = cal[pos_of[hold_starts[0]]:]
r = daily.loc[hd].to_numpy()
mo = (1 + pd.Series(r, index=hd)).groupby(hd.to_period("M")).prod() - 1
sharpe = r.mean() / r.std(ddof=1) * math.sqrt(252)
tm = mo.mean() / mo.std(ddof=1) * math.sqrt(len(mo))

mod = pd.read_csv(SCR / "holdout_trades_selected.csv", parse_dates=["entry", "exit"])
ho = json.loads((RES / "holdout.json").read_text())
print(f"independent: {len(mine)} trades, Sharpe {sharpe:.4f}, monthly t {tm:.3f}, mean/trade {mine['net'].mean() * 100:.4f}%")
print(f"module:      {len(mod)} trades, Sharpe {ho['summary']['sharpe']:.4f}, monthly t {ho['summary']['t_monthly']:.3f}, "
      f"mean/trade {mod['net'].mean() * 100:.4f}%")
key = ["formation", "pair", "entry", "exit"]
m = mine.merge(mod[key + ["net", "reason"]], on=key, how="outer", suffixes=("_ind", "_mod"), indicator=True)
print("trade match:", m["_merge"].value_counts().to_dict(),
      "| max |net diff|:", float((m["net_ind"] - m["net_mod"]).abs().max()),
      "| reasons agree:", bool((m.dropna()["reason_ind"] == m.dropna()["reason_mod"]).all()))
h1, h2 = mo[mo.index < pd.Period("2020-01", "M")].mean(), mo[mo.index >= pd.Period("2020-01", "M")].mean()
print(f"halves (mean monthly %): {h1 * 100:+.4f} / {h2 * 100:+.4f} vs module {ho['halves_mean_monthly_pct']}")

# one trade by hand from the raw rows
tr = mine.iloc[len(mine) // 2]
a, b = tr["pair"].split("/")
raw = df[df["ticker"].isin([a, b]) & df["date"].isin([tr["entry"], tr["exit"]])].set_index(["ticker", "date"])
adj_open = lambda t, d: raw.loc[(t, d), "open"] * raw.loc[(t, d), "closeadj"] / raw.loc[(t, d), "close"]
row = mod[(mod["pair"] == tr["pair"]) & (mod["entry"] == tr["entry"])].iloc[0]
long_, short_ = row["long"], row["short"]
px_exit = (lambda t: adj_open(t, tr["exit"])) if tr["reason"] in ("converged", "time") else \
    (lambda t: raw.loc[(t, tr["exit"]), "closeadj"])
rl = px_exit(long_) / adj_open(long_, tr["entry"]) - 1
rs = px_exit(short_) / adj_open(short_, tr["entry"]) - 1
sessions = pos_of[tr["exit"]] - pos_of[tr["entry"]]
by_hand = (rl - rs) / 2 - COST - sessions * BORROW
print(f"by hand: {tr['pair']} long {long_} {tr['entry'].date()} -> {tr['exit'].date()} ({tr['reason']}): "
      f"long leg {rl * 100:+.3f}%, short leg {rs * 100:+.3f}%, net {by_hand * 100:+.4f}% vs module {row['net'] * 100:+.4f}%")
