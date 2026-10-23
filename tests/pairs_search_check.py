"""
Checks for replay/pairs_search.py on synthetic data (no market data, no returns
of the real test): z-scores against pandas, the numba simulator against an
independently written reference for all 480 settings, the neighbour grid, pair
selection rules, and batch vs single runs.

    python tests/pairs_search_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay import pairs_search as ps  # noqa: E402

rng = np.random.default_rng(7)
BD = ps.BORROW_YR * 0.5 / 252


def check_zscores():
    for _ in range(20):
        n = int(rng.integers(5, 140))
        s = np.cumsum(rng.normal(0, 0.01, ps.FORM_WIN + n)) + 0.3
        Z = ps.zscores(s, ps.FORM_WIN, ps.LB_ARR)
        ser = pd.Series(s)
        for li, L in enumerate(ps.LOOKBACKS):
            if L == 0:
                f = ser.iloc[:ps.FORM_WIN]
                ref = ((ser - f.mean()) / f.std(ddof=1)).iloc[ps.FORM_WIN:].to_numpy()
            else:
                ref = ((ser - ser.rolling(L).mean()) / ser.rolling(L).std(ddof=1)).iloc[ps.FORM_WIN:].to_numpy()
            assert np.allclose(Z[li], ref, atol=1e-8), (L, np.abs(Z[li] - ref).max())
    print("zscores match pandas rolling (ddof=1) and the static formation window")


def reference(z, ca, cb, oa, ob, n, data_end, entry, target, stop, hold):
    """Independent version: find each trade's exit by walking forward, then mark P&L from the trade list."""
    trades, t, armed = [], 0, True
    while t < n:
        zt = z[t]
        if not np.isnan(zt):
            if not armed and abs(zt) < entry / 2:
                armed = True
            if armed and abs(zt) >= entry and t + 1 <= n - 2:
                pos, e, armed = (1 if zt < 0 else -1), t + 1, False
                d = e
                while True:
                    zd, held = z[d], d - e + 1
                    if d == n - 1:
                        reason, x, at_open = (5 if data_end else 4), d, False
                        break
                    if not np.isnan(zd) and pos * zd >= -target:
                        reason, x, at_open = 1, d + 1, True
                        break
                    if stop and not np.isnan(zd) and pos * zd <= -(entry + 2):
                        reason, x, at_open = 2, d + 1, True
                        break
                    if held >= hold:
                        reason, x, at_open = 3, d + 1, True
                        break
                    d += 1
                trades.append((e, x, pos, reason, at_open))
                if not at_open:
                    break
                t = x
                continue
        t += 1
    daily = np.zeros(n)
    nets = []
    for e, x, pos, reason, at_open in trades:
        wa, wb = 0.5 * pos, -0.5 * pos
        pa, pb = oa[e], ob[e]
        last = x - 1 if at_open else x
        tot = 0.0
        for d in range(e, last + 1):
            v = wa * (ca[d] - pa) / oa[e] + wb * (cb[d] - pb) / ob[e] - BD
            daily[d] += v
            tot += v
            pa, pb = ca[d], cb[d]
        if at_open:
            v = wa * (oa[x] - pa) / oa[e] + wb * (ob[x] - pb) / ob[e]
            daily[x] += v
            tot += v
        daily[e] -= ps.COST_RT / 2
        daily[x] -= ps.COST_RT / 2
        tot -= ps.COST_RT
        nets.append(tot)
        # telescoping check: the trade's net is the leg returns from entry open to exit price, less costs
        px = (oa[x], ob[x]) if at_open else (ca[x], cb[x])
        gross = wa * (px[0] / oa[e] - 1) + wb * (px[1] / ob[e] - 1)
        assert abs(tot - (gross - ps.COST_RT - (x - e + (0 if at_open else 1)) * BD)) < 1e-12
    return trades, daily, nets


def check_simulator(n_cases=150):
    ct = ps.config_table()
    cl, ce, ctg, cs, chold = ps.sub_arrays(ct)
    total_trades = 0
    reasons = set()
    for case in range(n_cases):
        n = int(rng.integers(3, 130))
        # two prices sharing a factor, plus a mean-reverting spread
        f = np.cumsum(rng.normal(0, 0.012, ps.FORM_WIN + n))
        u = np.zeros(ps.FORM_WIN + n)
        for i in range(1, len(u)):
            u[i] = 0.9 * u[i - 1] + rng.normal(0, 0.02)
        ca_all = 50 * np.exp(f + u)
        cb_all = 80 * np.exp(f)
        s = np.log(ca_all) - np.log(cb_all)
        s = s - s[:ps.FORM_WIN].mean()
        Z = ps.zscores(s, ps.FORM_WIN, ps.LB_ARR)
        Z[:, rng.random(n) < 0.05] = np.nan                       # odd missing z values
        ca, cb = ca_all[ps.FORM_WIN:], cb_all[ps.FORM_WIN:]
        oa = ca * np.exp(rng.normal(0, 0.004, n))
        ob = cb * np.exp(rng.normal(0, 0.004, n))
        data_end = bool(case % 3 == 0)
        for c in range(ps.SUB):
            daily = np.zeros(n + 5)
            buf = np.zeros((n + 2, 7))
            ntr, snet = ps.sim_one(Z[cl[c]], ca, cb, oa, ob, n, data_end, ce[c], ctg[c], cs[c], chold[c],
                                   ps.COST_RT, BD, daily, 2, 1.0, buf, True)
            tr, dref, nets = reference(Z[cl[c]], ca, cb, oa, ob, n, data_end, ce[c], ctg[c], cs[c], chold[c])
            assert ntr == len(tr), (case, c, ntr, len(tr))
            assert np.allclose(daily[2:2 + n], dref, atol=1e-12), (case, c)
            assert abs(daily[:2].sum()) == 0 and abs(daily[2 + n:].sum()) == 0
            for (e, x, pos, reason, at_open), r, net in zip(tr, buf[:ntr], nets):
                assert (int(r[0]), int(r[1]), int(r[2]), int(r[3]), bool(r[6])) == (e, x, pos, reason, at_open)
                assert abs(r[4] - net) < 1e-12
                reasons.add(reason)
            assert abs(snet - sum(nets)) < 1e-10
            total_trades += ntr
    assert reasons == {1, 2, 3, 4, 5}, reasons
    print(f"simulator matches the reference on {n_cases} pair-periods x {ps.SUB} settings "
          f"({total_trades:,} trades, every exit reason exercised)")


def check_grid():
    ct = ps.config_table()
    assert len(ct) == ps.NCFG == 960 and ct.iloc[:ps.SUB]["method"].eq(ps.METHODS[0]).all()
    nb = ps.neighbours(ct)
    sizes = pd.Series([len(x) for x in nb])
    # interior: lookback 2 + entry 2 + target 1 + stop 1 + hold 2 + method 1 = 9
    assert sizes.max() == 9 and sizes.min() == 6, sizes.value_counts()
    for i, x in enumerate(nb):
        for j in x:
            assert i in nb[j]
            diff = (ct.iloc[i][["_m", "_l", "_e", "_t", "_s", "_h"]] - ct.iloc[j][["_m", "_l", "_e", "_t", "_s", "_h"]]).abs()
            assert diff.sum() == 1
    sh = rng.normal(size=ps.NCFG)
    pl = ps.plateau(sh, nb)
    assert abs(pl[0] - np.mean(sh[[0] + nb[0]])) < 1e-12
    print("grid: 960 settings, neighbours symmetric and one step apart (6-9 each), plateau = local mean")


def toy_panel():
    pn = object.__new__(ps.Panel)
    T = ps.FORM_WIN + 200
    pn.cal = pd.bdate_range("2000-01-03", periods=T)
    names = [f"S{i:02d}" for i in range(12)]
    pn.tickers, pn.col = names, {t: i for i, t in enumerate(names)}
    base = np.cumsum(rng.normal(0, 0.01, (T, 12)), axis=0)
    base[:, 1] = base[:, 0] + rng.normal(0, 0.001, T)             # S00/S01 near-twins
    base[:, 3] = base[:, 2] + rng.normal(0, 0.002, T)             # S02/S03
    pn.C = 40 * np.exp(base)
    pn.O, pn.P = pn.C.copy(), pn.C.copy()
    pn.DV = np.tile(np.arange(12, 0, -1) * 1e6, (T, 1))
    pn.industry = {t: ("ind1" if i < 6 else "ind2") for i, t in enumerate(names)}
    pn.sector = {t: ("secA" if i % 2 else "secB") for i, t in enumerate(names)}
    pn.cname = {t: t for t in names}
    pn.cname["S04"] = pn.cname["S05"] = "SAME CO"               # share classes
    return pn


def check_pairs():
    pn = toy_panel()
    k = ps.FORM_WIN + 10
    top = pn.tickers
    for m in ps.METHODS:
        p = pn.pairs(k, top, m)
        flat = [x for ab in p for x in ab]
        assert len(flat) == len(set(flat)), "a stock used twice"
        assert all(pn.industry[a] == pn.industry[b] for a, b in p), "cross-industry pair"
        assert ("S04", "S05") not in p, "share classes paired"
        assert p[0] == ("S00", "S01"), (m, p)
    r = pn.pairs(k, top, "random", np.random.default_rng(1))
    flat = [x for ab in r for x in ab]
    assert len(flat) == len(set(flat)) and all(pn.sector[a] != pn.sector[b] for a, b in r)
    assert ps.norm_name("Alphabet Inc - Class A") == ps.norm_name("ALPHABET INC CLASS C")
    print("pairs: disjoint, same industry, share classes excluded, closest twin first; random pairs cross-sector")


def check_batch_vs_single():
    pn = toy_panel()
    ct = ps.config_table()
    subs = ps.sub_arrays(ct)
    forms = [(ps.FORM_WIN + 1, ps.FORM_WIN + 100), (ps.FORM_WIN + 100, ps.FORM_WIN + 200)]
    sets = [[pn.pairs(k, pn.tickers, m) for k, _ in forms] for m in ps.METHODS]
    day0, nd = forms[0][0], forms[-1][1] - forms[0][0]
    daily, ntr, snet, nform = ps.run_block(pn, forms, sets, subs, day0, nd)
    for c in rng.choice(ps.NCFG, 40, replace=False):
        m, sub = divmod(int(c), ps.SUB)
        d1, tr = ps.run_single(pn, forms, sets[m], sub, subs, day0, nd)
        assert np.allclose(daily[c], d1, atol=1e-12) and len(tr) == ntr[c] == nform[c].sum()
        if len(tr):
            assert abs(tr["net"].sum() - snet[c]) < 1e-10
    print("batch run equals single-setting runs (daily returns, trade counts, net sums)")


if __name__ == "__main__":
    check_zscores()
    check_grid()
    check_pairs()
    check_simulator()
    check_batch_vs_single()
    print("\nPAIRS SEARCH CHECKS OK")
