"""
Pairs trading: search 960 settings on 1999-2012, test the chosen one once on
2013-2026. Pre-registered 2026-10-23 (results/pairs_search/PREREGISTRATION.md).

Luis asked whether trying many settings and keeping the one that works can be
told apart from a backtest fooling itself. The method: search only on old data,
pick the winner by a rule fixed in advance (the best plateau of good neighbours,
not a lone spike), measure how good the best of 960 looks by luck (White's
Reality Check, the Deflated Sharpe Ratio, and the same search run on random
unrelated pairs), then look once at 2013-2026.

Pairs are formed point in time on the first session of every January and July
from the 300 most traded S&P 500 members: the 20 closest same-industry pairs by
normalized-price distance (S1) or daily-return correlation (S2), each stock used
once. They trade for six months under the setting's rule: the spread is
ln(A) - ln(B) and its z-score; when the gap reaches the entry level at a close,
the next open buys the cheap stock and shorts the rich one with equal dollars.

    python -m replay.pairs_search --data DIR --stage search    # 1999-2012 only
    python -m replay.pairs_search --data DIR --stage holdout   # needs selection.json
DIR holds sp500_members_daily_1997_2008.csv.gz, sp500_members_daily_2009_2026.csv.gz,
spy_daily.csv.gz, sp500_membership_raw.csv and sp500_tickers_industry.csv, derived
from Luis's Sharadar bulk files (licensed: not committed). Per-trade detail and
daily return arrays are written to --scratch, never to the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import kurtosis, norm, skew, spearmanr

from . import config
from .asof import sp500_members

OUT = config.RESULTS_DIR / "pairs_search"

# ── Locked by the pre-registration ───────────────────────────────────────────
LOOKBACKS = (20, 40, 60, 120, 250, 0)      # 0 = static: the formation window's mean and sd
ENTRIES = (1.0, 1.5, 2.0, 2.5, 3.0)
TARGETS = (0.0, 0.5)                       # exit once the gap is back within this many sd
STOPS = (0, 1)                             # 1 = stop when the gap widens to entry + 2 sd
HOLDS = (5, 10, 20, 60)                    # sessions
METHODS = ("S1_distance", "S2_correlation")
FORM_WIN, TOP_LIQ, DV_WIN, MIN_PRICE, N_PAIRS = 252, 300, 60, 5.0, 20
COST_RT, BORROW_YR = 0.0015, 0.005
SEARCH_YEARS, HOLDOUT_YEARS = (1999, 2012), (2013, 2026)
SPLIT = "2020-01-01"
MIN_TRADES = 300
R_NULL, RC_B, RC_BLOCK, SEED = 200, 2000, 20, 20261023
PASS = {"min_trades": 100, "t_monthly": 2.0, "drop_top_share": 0.03, "max_pair_share": 0.25}
EULER = 0.5772156649015329
SUB = len(LOOKBACKS) * len(ENTRIES) * len(TARGETS) * len(STOPS) * len(HOLDS)      # 480 per method
NCFG = SUB * len(METHODS)                                                            # 960
LB_ARR = np.array(LOOKBACKS, dtype=np.int64)
REASONS = {1: "converged", 2: "stop", 3: "time", 4: "period end", 5: "data end"}


# ─────────────────────────────────────────────────────────────────────────────
#  Settings grid
# ─────────────────────────────────────────────────────────────────────────────

def config_table() -> pd.DataFrame:
    rows = []
    for m, li, ei, ti, si, hi in itertools.product(range(len(METHODS)), range(len(LOOKBACKS)), range(len(ENTRIES)),
                                                 range(len(TARGETS)), range(len(STOPS)), range(len(HOLDS))):
        rows.append({"method": METHODS[m], "lookback": LOOKBACKS[li] if LOOKBACKS[li] else "static",
                     "entry_z": ENTRIES[ei], "exit_within": TARGETS[ti], "stop": "entry+2" if STOPS[si] else "none",
                     "max_hold": HOLDS[hi], "_m": m, "_l": li, "_e": ei, "_t": ti, "_s": si, "_h": hi})
    ct = pd.DataFrame(rows)
    ct.index.name = "config"
    return ct


def sub_arrays(ct: pd.DataFrame):
    s = ct.iloc[:SUB]
    return (s["_l"].to_numpy(np.int64), np.array([ENTRIES[i] for i in s["_e"]]),
            np.array([TARGETS[i] for i in s["_t"]]), s["_s"].to_numpy(np.int64),
            np.array([HOLDS[i] for i in s["_h"]], dtype=np.int64))


def neighbours(ct: pd.DataFrame) -> list[list[int]]:
    """Settings one step away in exactly one dimension (lookback order 20,40,60,120,250,static)."""
    dims = ["_m", "_l", "_e", "_t", "_s", "_h"]
    lim = [len(METHODS), len(LOOKBACKS), len(ENTRIES), len(TARGETS), len(STOPS), len(HOLDS)]
    key = {tuple(r): i for i, r in enumerate(ct[dims].to_numpy())}
    out = []
    for r in ct[dims].to_numpy():
        nb = set()
        for d in range(len(dims)):
            for step in (-1, 1):
                q = list(r)
                q[d] += step
                if 0 <= q[d] < lim[d]:
                    nb.add(key[tuple(q)])
        out.append(sorted(nb))
    return out


def plateau(sh: np.ndarray, nbrs: list[list[int]]) -> np.ndarray:
    return np.array([np.nanmean(sh[[i] + nb]) for i, nb in enumerate(nbrs)])


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation (numba)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def zscores(s, nform, lookbacks):
    """z of the spread at each trading-day close; s = formation window then trading window."""
    m = s.shape[0]
    n = m - nform
    Z = np.full((lookbacks.shape[0], n), np.nan)
    c1 = np.zeros(m + 1)
    c2 = np.zeros(m + 1)
    for i in range(m):
        c1[i + 1] = c1[i] + s[i]
        c2[i + 1] = c2[i] + s[i] * s[i]
    for li in range(lookbacks.shape[0]):
        L = lookbacks[li]
        if L == 0:
            mu = c1[nform] / nform
            var = (c2[nform] - nform * mu * mu) / (nform - 1)
            if var > 1e-18:
                sd = math.sqrt(var)
                for t in range(n):
                    Z[li, t] = (s[nform + t] - mu) / sd
        else:
            for t in range(n):
                i = nform + t
                a = i - L + 1
                mu = (c1[i + 1] - c1[a]) / L
                var = (c2[i + 1] - c2[a] - L * mu * mu) / (L - 1)
                if var > 1e-18:
                    Z[li, t] = (s[i] - mu) / math.sqrt(var)
    return Z


@njit(cache=True)
def sim_one(z, ca, cb, oa, ob, n, data_end, entry, target, stop, hold, cost_rt, borrow_d,
            daily, off, scale, trades, tlog):
    """One setting on one pair for one trading period. Day 0 = formation day.
    Signal at the close of t, trade at the open of t+1. Returns (trades, sum of net)."""
    ch = cost_rt * 0.5
    pos = 0
    armed = True
    e = -1
    pend = 0
    wa = 0.0
    wb = 0.0
    oae = 1.0
    obe = 1.0
    tnet = 0.0
    tgr = 0.0
    ntr = 0
    snet = 0.0
    for t in range(n):
        if pend != 0:                                   # exit at today's open
            g = wa * (oa[t] - ca[t - 1]) / oae + wb * (ob[t] - cb[t - 1]) / obe
            p = g - ch
            daily[off + t] += p * scale
            tnet += p
            tgr += g
            if tlog and ntr < trades.shape[0]:
                trades[ntr, 0] = e
                trades[ntr, 1] = t
                trades[ntr, 2] = pos
                trades[ntr, 3] = pend
                trades[ntr, 4] = tnet
                trades[ntr, 5] = tgr
                trades[ntr, 6] = 1.0
            ntr += 1
            snet += tnet
            pos = 0
            pend = 0
            armed = False
        if pos != 0:
            if t == e:
                g = wa * (ca[t] - oae) / oae + wb * (cb[t] - obe) / obe
                p = g - ch - borrow_d
            else:
                g = wa * (ca[t] - ca[t - 1]) / oae + wb * (cb[t] - cb[t - 1]) / obe
                p = g - borrow_d
            daily[off + t] += p * scale
            tnet += p
            tgr += g
            zt = z[t]
            held = t - e + 1
            r = 0
            if t == n - 1:
                r = 5 if data_end else 4
            elif zt == zt and pos * zt >= -target:
                r = 1
            elif stop == 1 and zt == zt and pos * zt <= -(entry + 2.0):
                r = 2
            elif held >= hold:
                r = 3
            if r >= 4:                                  # out at this close
                daily[off + t] -= ch * scale
                tnet -= ch
                if tlog and ntr < trades.shape[0]:
                    trades[ntr, 0] = e
                    trades[ntr, 1] = t
                    trades[ntr, 2] = pos
                    trades[ntr, 3] = r
                    trades[ntr, 4] = tnet
                    trades[ntr, 5] = tgr
                    trades[ntr, 6] = 0.0
                ntr += 1
                snet += tnet
                pos = 0
                armed = False
            elif r > 0:
                pend = r
            continue
        zt = z[t]
        if zt != zt:
            continue
        az = abs(zt)
        if not armed and az < entry * 0.5:
            armed = True
        if armed and az >= entry and t + 1 <= n - 2:
            pos = 1 if zt < 0 else -1                   # z < 0: A cheap -> long A, short B
            e = t + 1
            wa = 0.5 * pos
            wb = -0.5 * pos
            oae = oa[e]
            obe = ob[e]
            tnet = 0.0
            tgr = 0.0
            armed = False
    return ntr, snet


@njit(cache=True)
def run_configs(Z, ca, cb, oa, ob, n, data_end, cl, ce, ctg, cs, chold, cost_rt, borrow_d,
                daily, off, scale, ntr_out, snet_out, ntr_form, fi):
    dummy = np.zeros((1, 7))
    for c in range(cl.shape[0]):
        k, s = sim_one(Z[cl[c]], ca, cb, oa, ob, n, data_end, ce[c], ctg[c], cs[c], chold[c], cost_rt, borrow_d,
                       daily[c], off, scale, dummy, False)
        ntr_out[c] += k
        snet_out[c] += s
        ntr_form[c, fi] += k


@njit(cache=True)
def stationary_idx(T, p, u_jump, starts, first):
    idx = np.empty(T, np.int64)
    idx[0] = first
    for t in range(1, T):
        idx[t] = starts[t] if u_jump[t] < p else (idx[t - 1] + 1) % T
    return idx


# ─────────────────────────────────────────────────────────────────────────────
#  Data, universe and pairs
# ─────────────────────────────────────────────────────────────────────────────

class Panel:
    def __init__(self, d: Path):
        t0 = time.time()
        a = pd.read_csv(d / "sp500_members_daily_1997_2008.csv.gz", parse_dates=["date"])
        b = pd.read_csv(d / "sp500_members_daily_2009_2026.csv.gz", parse_dates=["date"])
        df = pd.concat([a, b], ignore_index=True)
        spy = pd.read_csv(d / "spy_daily.csv.gz", parse_dates=["date"]).set_index("date").sort_index()
        self.cal = spy.loc[df["date"].min():df["date"].max()].index

        def piv(col):
            return df.pivot(index="date", columns="ticker", values=col).reindex(self.cal)
        C, close = piv("closeadj"), piv("close")
        self.tickers = list(C.columns)
        self.col = {t: i for i, t in enumerate(self.tickers)}
        self.C = C.to_numpy(np.float64)
        self.O = (piv("open") * C / close).to_numpy(np.float64)
        self.P = piv("closeunadj").to_numpy(np.float64)
        self.DV = (close * piv("volume")).to_numpy(np.float64)
        self.sp = pd.read_csv(d / "sp500_membership_raw.csv", parse_dates=["date"])
        tk = pd.read_csv(d / "sp500_tickers_industry.csv").set_index("ticker")
        self.industry = tk["industry"].fillna(tk["sicindustry"]).to_dict()
        self.sector = tk["sector"].dropna().to_dict()
        self.cname = {t: norm_name(v) for t, v in tk["name"].items()}
        print(f"[data] {len(df):,} rows, {len(self.tickers)} tickers, {len(self.cal)} sessions "
              f"{self.cal[0].date()}..{self.cal[-1].date()} ({time.time() - t0:.0f}s)", flush=True)

    def formations(self, years) -> list[tuple[int, int]]:
        ks = []
        for y in range(1999, HOLDOUT_YEARS[1] + 1):
            for mth in (1, 7):
                k = self.cal.searchsorted(pd.Timestamp(f"{y}-{mth:02d}-01"))
                if k < len(self.cal):
                    ks.append(k)
        out = []
        for i, k in enumerate(ks):
            k2 = ks[i + 1] if i + 1 < len(ks) else len(self.cal)
            if years[0] <= self.cal[k].year <= years[1]:
                out.append((k, k2))
        return out

    def universe(self, k: int) -> list[str]:
        prev = self.cal[k - 1]
        mem = [t for t in sp500_members(self.sp, prev.date()) if t in self.col]
        idx = np.array([self.col[t] for t in mem])
        W = self.C[k - FORM_WIN:k, idx]
        ok = ~np.isnan(W).any(axis=0) & (self.P[k - 1, idx] >= MIN_PRICE)
        mem, idx = [t for t, o in zip(mem, ok) if o], idx[ok]
        with np.errstate(all="ignore"):
            dv = np.nanmean(self.DV[k - DV_WIN:k, idx], axis=0)
        order = sorted(range(len(mem)), key=lambda i: (-np.nan_to_num(dv[i], nan=-1.0), mem[i]))
        return [mem[i] for i in order[:TOP_LIQ]]

    def pairs(self, k: int, top: list[str], method: str, rng=None) -> list[tuple[str, str]]:
        if method == "random":
            pool = [t for t in top if t in self.sector]
            order = [pool[i] for i in rng.permutation(len(pool))]
            used, out = set(), []
            for i, a in enumerate(order):
                if a in used:
                    continue
                for b in order[i + 1:]:
                    if b not in used and self.sector[b] != self.sector[a]:
                        out.append(tuple(sorted((a, b))))
                        used |= {a, b}
                        break
                if len(out) == N_PAIRS:
                    break
            return out
        X = self.C[k - FORM_WIN:k, [self.col[t] for t in top]]
        if method == "S1_distance":
            Nm = X / X[0]
        else:
            R = np.corrcoef(np.diff(np.log(X), axis=0).T)
        groups: dict[str, list[int]] = {}
        for i, t in enumerate(top):
            g = self.industry.get(t)
            if isinstance(g, str) and g:
                groups.setdefault(g, []).append(i)
        cand = []
        for g, ix in groups.items():
            for x in range(len(ix)):
                for y in range(x + 1, len(ix)):
                    i, j = ix[x], ix[y]
                    if self.cname.get(top[i]) and self.cname.get(top[i]) == self.cname.get(top[j]):
                        continue                         # share classes of one company
                    sc = float(((Nm[:, i] - Nm[:, j]) ** 2).sum()) if method == "S1_distance" else -float(R[i, j])
                    if not math.isfinite(sc):
                        continue
                    a, b = sorted((top[i], top[j]))
                    cand.append((sc, a, b))
        cand.sort()
        used, out = set(), []
        for sc, a, b in cand:
            if a in used or b in used:
                continue
            out.append((a, b))
            used |= {a, b}
            if len(out) == N_PAIRS:
                break
        return out

    def pair_arrays(self, k: int, k2: int, a: str, b: str):
        ia, ib = self.col[a], self.col[b]
        ca = self.C[k - FORM_WIN:k2, ia].copy()
        cb = self.C[k - FORM_WIN:k2, ib].copy()
        ntot = k2 - k
        valid = ~np.isnan(ca[FORM_WIN:]) & ~np.isnan(cb[FORM_WIN:])
        if not valid.any():
            return None
        n = int(np.nonzero(valid)[0][-1]) + 1
        ca, cb = ffill(ca[:FORM_WIN + n]), ffill(cb[:FORM_WIN + n])
        oa = self.O[k:k + n, ia].copy()
        ob = self.O[k:k + n, ib].copy()
        prev_a, prev_b = ca[FORM_WIN - 1:FORM_WIN - 1 + n], cb[FORM_WIN - 1:FORM_WIN - 1 + n]
        oa = np.where(np.isnan(oa), prev_a, oa)
        ob = np.where(np.isnan(ob), prev_b, ob)
        s = np.log(ca) - np.log(cb)
        s = s - s[:FORM_WIN].mean()
        Z = zscores(s, FORM_WIN, LB_ARR)
        return Z, ca[FORM_WIN:], cb[FORM_WIN:], oa, ob, n, n < ntot


def ffill(x: np.ndarray) -> np.ndarray:
    m = np.isnan(x)
    if not m.any():
        return x
    idx = np.where(~m, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def norm_name(s) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(r"\b(CLASS|CL|SERIES|SER)\s+[A-Z]\b", " ", s)
    return " ".join(s.split())


# ─────────────────────────────────────────────────────────────────────────────
#  Running blocks of formations
# ─────────────────────────────────────────────────────────────────────────────

def run_block(pn: Panel, forms, sets, subs, day0: int, ndays: int, cost_rt=COST_RT):
    """sets[m][fi] = pairs for method m at formation fi. Returns daily (len(sets)*480, ndays), trades, net sums,
    trades per formation."""
    cl, ce, ctg, cs, chold = subs
    K = len(sets) * SUB
    daily = np.zeros((K, ndays))
    ntr, snet = np.zeros(K), np.zeros(K)
    nform = np.zeros((K, len(forms)))
    bd = BORROW_YR * 0.5 / 252
    for m in range(len(sets)):
        rows = slice(m * SUB, (m + 1) * SUB)
        dv, nv, sv, fv = daily[rows], ntr[rows], snet[rows], nform[rows]
        for fi, (k, k2) in enumerate(forms):
            for a, b in sets[m][fi]:
                arr = pn.pair_arrays(k, k2, a, b)
                if arr is None:
                    continue
                Z, ca, cb, oa, ob, n, dend = arr
                run_configs(Z, ca, cb, oa, ob, n, dend, cl, ce, ctg, cs, chold, cost_rt, bd,
                            dv, k - day0, 1.0 / N_PAIRS, nv, sv, fv, fi)
    return daily, ntr, snet, nform


def run_single(pn: Panel, forms, pairsets, sub_idx: int, subs, day0: int, ndays: int, log=True):
    """One setting, with a trade log. pairsets[fi] = pairs."""
    cl, ce, ctg, cs, chold = subs
    bd = BORROW_YR * 0.5 / 252
    daily = np.zeros(ndays)
    out = []
    for fi, (k, k2) in enumerate(forms):
        for a, b in pairsets[fi]:
            arr = pn.pair_arrays(k, k2, a, b)
            if arr is None:
                continue
            Z, ca, cb, oa, ob, n, dend = arr
            buf = np.zeros((n + 2, 7))
            ntr, _ = sim_one(Z[cl[sub_idx]], ca, cb, oa, ob, n, dend, ce[sub_idx], ctg[sub_idx], cs[sub_idx],
                             chold[sub_idx], COST_RT, bd, daily, k - day0, 1.0 / N_PAIRS, buf, True)
            for r in buf[:ntr]:
                e, x, pos = int(r[0]), int(r[1]), int(r[2])
                out.append({"formation": str(pn.cal[k].date()), "pair": f"{a}/{b}",
                            "long": a if pos > 0 else b, "short": b if pos > 0 else a,
                            "entry": pn.cal[k + e], "exit": pn.cal[k + x], "exit_at": "open" if r[6] else "close",
                            "sessions": x - e, "reason": REASONS[int(r[3])], "gross": r[5], "net": r[4]})
    return daily, pd.DataFrame(out)


def sharpe(daily: np.ndarray) -> np.ndarray:
    mu, sd = daily.mean(axis=-1), daily.std(axis=-1, ddof=1)
    with np.errstate(all="ignore"):
        return np.where(sd > 0, mu / sd * math.sqrt(252), np.nan)


def reality_check(R: np.ndarray, B: int, block: int, rng) -> dict:
    """White's Reality Check, studentized, stationary bootstrap. R: (T, K) daily returns."""
    T = R.shape[0]
    mu, sd = R.mean(0), R.std(0, ddof=1)
    ok = sd > 0
    R, mu, sd = R[:, ok], mu[ok], sd[ok]
    V = float(np.max(math.sqrt(T) * mu / sd))
    Vs = np.empty(B)
    for b0 in range(0, B, 100):
        nb = min(100, B - b0)
        Wt = np.zeros((nb, T))
        for j in range(nb):
            idx = stationary_idx(T, 1.0 / block, rng.random(T), rng.integers(0, T, T), int(rng.integers(0, T)))
            Wt[j] = np.bincount(idx, minlength=T)
        mstar = Wt @ R / T
        Vs[b0:b0 + nb] = np.max(math.sqrt(T) * (mstar - mu) / sd, axis=1)
    return {"statistic": V, "p_value": float((1 + (Vs >= V).sum()) / (B + 1)), "settings": int(ok.sum()),
            "B": B, "mean_block": block}


def dsr(r: np.ndarray, sr_all: np.ndarray, N: int) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), daily units."""
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    v = float(np.nanvar(sr_all, ddof=1))
    sr0 = math.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    return {"dsr": float(norm.cdf(z)), "sharpe_annual": float(sr * math.sqrt(252)),
            "hurdle_sharpe_annual": float(sr0 * math.sqrt(252)), "trials": N, "skew": g3, "kurtosis": g4}


def monthly(daily: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    return (1 + pd.Series(daily, index=dates)).groupby(dates.to_period("M")).prod() - 1


def max_drawdown(daily: np.ndarray) -> float:
    eq = np.cumprod(1 + daily)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def summarize(daily: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    mo = monthly(daily, dates)
    yr = (1 + pd.Series(daily, index=dates)).groupby(dates.year).prod() - 1
    t = float(mo.mean() / mo.std(ddof=1) * math.sqrt(len(mo))) if mo.std(ddof=1) > 0 else float("nan")
    return {"sharpe": float(sharpe(daily)), "annual_return_pct": float((np.prod(1 + daily) ** (252 / len(daily)) - 1) * 100),
            "months": int(len(mo)), "mean_monthly_pct": float(mo.mean() * 100), "t_monthly": t,
            "max_drawdown_pct": max_drawdown(daily) * 100,
            "by_year_pct": {int(k): round(float(v) * 100, 2) for k, v in yr.items()}}


# ─────────────────────────────────────────────────────────────────────────────
#  Stages
# ─────────────────────────────────────────────────────────────────────────────

def form_sets(pn: Panel, forms, methods, rng=None, universes=None):
    sets = []
    for m in methods:
        sets.append([pn.pairs(k, universes[fi] if universes else pn.universe(k), m, rng)
                     for fi, (k, _) in enumerate(forms)])
    return sets


def stage_search(pn: Panel, scratch: Path):
    ct, nb = config_table(), None
    nb = neighbours(ct)
    subs = sub_arrays(ct)
    forms = pn.formations(SEARCH_YEARS)
    day0, ndays = forms[0][0], forms[-1][1] - forms[0][0]
    dates = pn.cal[day0:day0 + ndays]
    t0 = time.time()
    unis = [pn.universe(k) for k, _ in forms]
    sets = form_sets(pn, forms, METHODS, universes=unis)
    daily, ntr, snet, nform = run_block(pn, forms, sets, subs, day0, ndays)
    sh = sharpe(daily)
    pl = plateau(sh, nb)
    elig = (ntr >= MIN_TRADES) & np.isfinite(sh)
    sel = int(np.lexsort((-np.nan_to_num(sh, nan=-9), -np.where(elig, pl, -9)))[0])
    raw = int(np.argmax(np.where(elig, sh, -9)))
    print(f"[search] {len(forms)} formations, {ndays} sessions, {int(elig.sum())} eligible settings "
          f"({time.time() - t0:.0f}s)", flush=True)

    res = ct.drop(columns=[c for c in ct.columns if c.startswith("_")]).copy()
    res["trades"] = ntr.astype(int)
    res["mean_net_per_trade_pct"] = np.where(ntr > 0, snet / np.maximum(ntr, 1) * 100, np.nan)
    res["sharpe"] = sh
    res["annual_return_pct"] = (np.prod(1 + daily, axis=1) ** (252 / ndays) - 1) * 100
    res["plateau"] = pl
    res["eligible"] = elig

    # 2x cost sensitivity (reported only)
    d2, n2, _, _ = run_block(pn, forms, sets, subs, day0, ndays, cost_rt=2 * COST_RT)
    sh2 = sharpe(d2)
    pl2 = plateau(sh2, nb)
    res["sharpe_2x_cost"] = sh2
    el2 = (n2 >= MIN_TRADES) & np.isfinite(sh2)
    sel2 = int(np.lexsort((-np.nan_to_num(sh2, nan=-9), -np.where(el2, pl2, -9)))[0])

    # how good the best of 960 looks by luck
    rng = np.random.default_rng(SEED)
    rc = reality_check(daily.T.copy(), RC_B, RC_BLOCK, rng)
    srd = np.where(np.isfinite(sh), sh / math.sqrt(252), np.nan)
    d_sel = dsr(daily[sel], srd, NCFG)
    d_raw = dsr(daily[raw], srd, NCFG)
    t1 = time.time()
    null_best_sh, null_best_pl = np.empty(R_NULL), np.empty(R_NULL)
    for r in range(R_NULL):
        rs = form_sets(pn, forms, ["random", "random"], rng=rng, universes=unis)
        dn, nn, _, _ = run_block(pn, forms, rs, subs, day0, ndays)
        shn = sharpe(dn)
        en = (nn >= MIN_TRADES) & np.isfinite(shn)
        null_best_sh[r] = np.max(np.where(en, shn, -9))
        null_best_pl[r] = np.max(np.where(en, plateau(shn, nb), -9))
        if (r + 1) % 20 == 0:
            print(f"[null] {r + 1}/{R_NULL} ({time.time() - t1:.0f}s)", flush=True)
    rn = {"repetitions": R_NULL,
          "real_best_sharpe": float(sh[raw]), "real_best_plateau": float(pl[sel]),
          "null_best_sharpe_median": float(np.median(null_best_sh)),
          "null_best_sharpe_95pct": float(np.percentile(null_best_sh, 95)),
          "null_best_plateau_median": float(np.median(null_best_pl)),
          "null_best_plateau_95pct": float(np.percentile(null_best_pl, 95)),
          "p_best_sharpe": float((1 + (null_best_sh >= sh[raw]).sum()) / (R_NULL + 1)),
          "p_best_plateau": float((1 + (null_best_pl >= pl[sel]).sum()) / (R_NULL + 1))}

    def cfg(i):
        r = res.loc[i]
        return {"config": int(i), **{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}}
    selection = {
        "stage": "search", "search_years": SEARCH_YEARS, "sessions": ndays,
        "first_day": str(dates[0].date()), "last_day": str(dates[-1].date()),
        "eligible_settings": int(elig.sum()), "selected": cfg(sel), "raw_best": cfg(raw),
        "selected_neighbours": [cfg(i) for i in nb[sel]],
        "selected_summary": summarize(daily[sel], dates),
        "share_of_settings_with_positive_sharpe": float(np.mean(sh[np.isfinite(sh)] > 0)),
        "median_sharpe": float(np.nanmedian(sh)),
        "reality_check": rc, "deflated_sharpe_selected": d_sel, "deflated_sharpe_raw_best": d_raw,
        "random_pair_null": rn,
        "selection_at_2x_cost": {"config": sel2, "same_as_selected": sel2 == sel,
                                 "selected_sharpe_at_2x_cost": float(sh2[sel])},
        "pairs_formed": {str(pn.cal[k].date()): {m: [f"{a}/{b}" for a, b in sets[mi][fi]]
                                                 for mi, m in enumerate(METHODS)}
                         for fi, (k, _) in enumerate(forms)}}
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "search_results.csv", float_format="%.6g")
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    np.savez_compressed(scratch / "search_daily.npz", daily=daily, nform=nform, day0=day0)
    s = selection["selected"]
    print(f"[search] selected #{sel}: {s['method']} lookback {s['lookback']} entry {s['entry_z']} exit within "
          f"{s['exit_within']} stop {s['stop']} hold {s['max_hold']} | Sharpe {s['sharpe']:.2f} plateau "
          f"{s['plateau']:.2f} trades {s['trades']}", flush=True)
    r = selection["raw_best"]
    print(f"[search] raw best #{raw}: Sharpe {r['sharpe']:.2f} trades {r['trades']} | RC p {rc['p_value']:.3f} | "
          f"DSR selected {d_sel['dsr']:.3f} | null p (plateau) {rn['p_best_plateau']:.3f} (sharpe) "
          f"{rn['p_best_sharpe']:.3f}", flush=True)


def stage_holdout(pn: Panel, scratch: Path):
    selp = OUT / "selection.json"
    sel_json = json.loads(selp.read_text())
    sel, raw = sel_json["selected"]["config"], sel_json["raw_best"]["config"]
    ct = config_table()
    nb = neighbours(ct)
    subs = sub_arrays(ct)
    forms = pn.formations(HOLDOUT_YEARS)
    day0, ndays = forms[0][0], forms[-1][1] - forms[0][0]
    dates = pn.cal[day0:day0 + ndays]
    unis = [pn.universe(k) for k, _ in forms]
    sets = form_sets(pn, forms, METHODS, universes=unis)

    # the one look: the selected setting
    m, sub = divmod(sel, SUB)
    d_sel, trades = run_single(pn, forms, sets[m], sub, subs, day0, ndays)
    summ = summarize(d_sel, dates)
    mo = monthly(d_sel, dates)
    h1, h2 = mo[mo.index < pd.Period(SPLIT[:7], "M")], mo[mo.index >= pd.Period(SPLIT[:7], "M")]
    net = trades["net"] if len(trades) else pd.Series(dtype=float)
    k = max(1, math.ceil(PASS["drop_top_share"] * len(net)))
    drop = float(net.sort_values().iloc[:-k].mean()) if len(net) > k else float("nan")
    per = trades.groupby("pair")["net"].sum() if len(trades) else pd.Series(dtype=float)
    share = float(per.abs().max() / per.abs().sum()) if len(per) and per.abs().sum() > 0 else 1.0
    crit = {"1_trades_ge_100": bool(len(trades) >= PASS["min_trades"]),
            "2_monthly_mean_pos_and_t_ge_2": bool(summ["mean_monthly_pct"] > 0 and summ["t_monthly"] >= PASS["t_monthly"]),
            "3_both_halves_pos": bool(len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0),
            "4_trade_mean_pos_without_top_3pct": bool(drop > 0),
            "5_no_pair_over_25pct": bool(share <= PASS["max_pair_share"])}
    verdict = {"stage": "holdout", "selection_sha256": hashlib.sha256(selp.read_bytes()).hexdigest(),
               "holdout_years": HOLDOUT_YEARS, "first_day": str(dates[0].date()), "last_day": str(dates[-1].date()),
               "selected": sel_json["selected"], "summary": summ,
               "trades": int(len(trades)), "mean_net_per_trade_pct": float(net.mean() * 100) if len(net) else None,
               "win_rate_pct": float((net > 0).mean() * 100) if len(net) else None,
               "halves_mean_monthly_pct": [float(h1.mean() * 100), float(h2.mean() * 100)],
               "drop_top": {"k": k, "mean_net_per_trade_pct": drop * 100},
               "max_pair_share": share, "top_pair": per.abs().idxmax() if len(per) else None,
               "exits": trades["reason"].value_counts().to_dict() if len(trades) else {},
               "criteria": crit, "verdict": "pass" if all(crit.values()) else "fail"}

    # secondary, after the verdict is fixed
    daily, ntr, _, nform = run_block(pn, forms, sets, subs, day0, ndays)
    assert np.allclose(daily[sel], d_sel, atol=1e-12), "batch and single runs disagree"
    sh_h = sharpe(daily)
    res = pd.read_csv(OUT / "search_results.csv", index_col="config")
    allc = res[["method", "lookback", "entry_z", "exit_within", "stop", "max_hold", "trades", "sharpe", "plateau",
                "eligible"]].rename(columns={"trades": "search_trades", "sharpe": "search_sharpe",
                                             "plateau": "search_plateau"})
    allc["holdout_trades"] = ntr.astype(int)
    allc["holdout_sharpe"] = sh_h
    el = allc["eligible"].astype(bool)
    rho = spearmanr(allc.loc[el, "search_sharpe"], allc.loc[el, "holdout_sharpe"], nan_policy="omit")
    dec = pd.qcut(allc.loc[el, "search_sharpe"].rank(method="first"), 10, labels=False) + 1
    by_dec = allc.loc[el].groupby(dec)[["search_sharpe", "holdout_sharpe"]].mean()
    m_r, sub_r = divmod(raw, SUB)
    d_raw, tr_raw = run_single(pn, forms, sets[m_r], sub_r, subs, day0, ndays)

    # walk-forward: re-pick every January by the same rule on all periods so far
    sp = np.load(scratch / "search_daily.npz")
    full = np.concatenate([sp["daily"], daily], axis=1)
    fdates = pn.cal[int(sp["day0"]):int(sp["day0"]) + full.shape[1]]
    nf_all = np.concatenate([sp["nform"], nform], axis=1)
    fstart = [pn.cal[k] for k, _ in pn.formations(SEARCH_YEARS)] + [pn.cal[k] for k, _ in forms]
    wf, picks = np.zeros(ndays), {}
    for y in range(HOLDOUT_YEARS[0], HOLDOUT_YEARS[1] + 1):
        cut = pd.Timestamp(f"{y}-01-01")
        w = fdates < cut
        nf = nf_all[:, [i for i, f in enumerate(fstart) if f < cut]].sum(axis=1)
        shw = sharpe(full[:, w])
        ok = (nf >= MIN_TRADES) & np.isfinite(shw)
        plw = plateau(shw, nb)
        c = int(np.lexsort((-np.nan_to_num(shw, nan=-9), -np.where(ok, plw, -9)))[0])
        picks[y] = c
        yy = dates.year == y
        wf[yy] = daily[c, yy]

    # the selected setting on random unrelated pairs in the holdout
    rng = np.random.default_rng(SEED + 1)
    null_h = np.empty(R_NULL)
    for r in range(R_NULL):
        rs = [pn.pairs(k, unis[fi], "random", rng) for fi, (k, _) in enumerate(forms)]
        dn, _ = run_single(pn, forms, rs, sub, subs, day0, ndays, log=False)
        null_h[r] = sharpe(dn)

    secondary = {
        "all_settings": {"spearman_search_vs_holdout_sharpe": float(rho.statistic), "p": float(rho.pvalue),
                         "eligible": int(el.sum()),
                         "share_positive_holdout_sharpe": float((allc.loc[el, "holdout_sharpe"] > 0).mean()),
                         "by_search_decile": {int(i): {"search_sharpe": float(r_["search_sharpe"]),
                                                       "holdout_sharpe": float(r_["holdout_sharpe"])}
                                              for i, r_ in by_dec.iterrows()}},
        "raw_best_holdout": {"config": raw, "summary": summarize(d_raw, dates), "trades": int(len(tr_raw)),
                             "mean_net_per_trade_pct": float(tr_raw["net"].mean() * 100) if len(tr_raw) else None},
        "walk_forward": {"summary": summarize(wf, dates), "picks": {int(y): int(c) for y, c in picks.items()},
                         "picks_described": {int(y): allc.loc[c, ["method", "lookback", "entry_z", "exit_within",
                                                                    "stop", "max_hold"]].to_dict()
                                             for y, c in picks.items()}},
        "selected_on_random_pairs": {"repetitions": R_NULL, "real_holdout_sharpe": summ["sharpe"],
                                     "random_median": float(np.median(null_h)),
                                     "random_95pct": float(np.percentile(null_h, 95)),
                                     "share_random_at_or_above_real": float((null_h >= summ["sharpe"]).mean())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "holdout.json").write_text(json.dumps({**verdict, "secondary": secondary}, indent=2, default=str))
    allc.to_csv(OUT / "holdout_all_settings.csv", float_format="%.6g")
    trades.to_csv(scratch / "holdout_trades_selected.csv", index=False)
    np.savez_compressed(scratch / "holdout_daily.npz", daily=daily, selected=d_sel, walk_forward=wf, day0=day0)
    print(f"[holdout] verdict {verdict['verdict']} | Sharpe {summ['sharpe']:.2f} t {summ['t_monthly']:.2f} "
          f"trades {len(trades)} | criteria {crit}", flush=True)
    print(f"[holdout] secondary: spearman {rho.statistic:.3f} | raw best Sharpe "
          f"{secondary['raw_best_holdout']['summary']['sharpe']:.2f} | walk-forward Sharpe "
          f"{secondary['walk_forward']['summary']['sharpe']:.2f} | random-pair median {np.median(null_h):.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", choices=["search", "holdout", "plumbing"], required=True)
    ap.add_argument("--scratch", default=".")
    a = ap.parse_args()
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    pn = Panel(Path(a.data))
    if a.stage == "plumbing":                      # pairs and data handling only; prints no returns
        for years in (SEARCH_YEARS, HOLDOUT_YEARS):
            forms = pn.formations(years)
            ns, ends = [], 0
            for k, k2 in forms[:: max(1, len(forms) // 4)]:
                top = pn.universe(k)
                for m in METHODS:
                    ps = pn.pairs(k, top, m)
                    print(f"[plumbing] {pn.cal[k].date()} {m}: {len(ps)} pairs, e.g. {ps[:5]}")
                    for a_, b_ in ps:
                        arr = pn.pair_arrays(k, k2, a_, b_)
                        ns.append(arr[5] if arr else 0)
                        ends += bool(arr and arr[6])
            print(f"[plumbing] {years}: {len(forms)} formations, sessions per pair-period "
                  f"{np.min(ns)}..{np.max(ns)}, cut short by data end: {ends}")
        return
    if a.stage == "search":
        stage_search(pn, scratch)
    else:
        if not (OUT / "selection.json").exists():
            raise SystemExit("selection.json missing: run and commit the search stage first")
        stage_holdout(pn, scratch)


if __name__ == "__main__":
    main()
