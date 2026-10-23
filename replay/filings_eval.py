"""
Evaluation for the SEC filings test (results/filings/PREREGISTRATION.md).

    python -m replay.filings_eval signals  --panel <dir>      # filing + monthly signals, no returns
    python -m replay.filings_eval search   --panel <dir>      # 8 settings on 2007-02..2018-12, selection
    python -m replay.filings_eval holdout  --panel <dir>      # one look at 2019-01..2026-09

<dir> holds panel_monthly.csv.gz and ticker_life.csv built on Luis's PC by
replay/filings_panel.py from Sharadar (licensed: never committed). Committed
outputs are aggregates only (monthly strategy returns, statistics); per-stock
files are written to <dir> as *_PRIVATE.csv.

The holdout stage refuses to run unless results/filings/selection.json is
committed and unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "filings"
LLM = RES / "llm"

SIGNALS = ["F1", "F2", "F3", "F4"]
HOLDS = [1, 3]
SEARCH = ("2007-02", "2018-12")            # return months
HOLDOUT = ("2019-01", "2026-09")
HALVES = (("2019-01", "2022-11"), ("2022-12", "2026-09"))
MAX_AGE_DAYS = 91
MIN_PRICE = 5.0
MIN_SECTOR = 10
QUANT = 0.2
COST_SIDE = 0.00075                         # 0.15% round trip
BORROW_M = 0.005 / 12
F4_GATE = 0.4
REF_DAYS, REF_MIN = 365, 50
RC_B, RC_BLOCK, N_PLACEBO, SEED = 5000, 6, 500, 20261023
NW_LAGS = 3
EULER = 0.5772156649015329


# ─────────────────────────────────────────────────────────────────────────────
#  Filing-level signals
# ─────────────────────────────────────────────────────────────────────────────

def accept_date(s: pd.Series) -> pd.Series:
    """EDGAR acceptanceDateTime carries a 'Z' but is New York wall-clock time
    (checked in the plumbing run: see the Amendments section)."""
    return pd.to_datetime(s.str[:10], errors="coerce")


def ref_percentile(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile (midrank) of each filing's value among filings of the same
    form accepted in the 365 days up to and including its acceptance date; if
    fewer than 50, among all earlier filings of that form."""
    out = pd.Series(np.nan, index=df.index)
    for form, g in df[df[col].notna()].groupby("form"):
        g = g.sort_values("adate")
        d = g["adate"].values.astype("datetime64[D]").astype(np.int64)
        v = g[col].values.astype(float)
        hi = np.searchsorted(d, d, side="right")
        lo = np.searchsorted(d, d - REF_DAYS, side="left")
        res = np.empty(len(g))
        for i in range(len(g)):
            a = lo[i] if hi[i] - lo[i] >= REF_MIN else 0
            ref = v[a:hi[i]]
            res[i] = ((ref < v[i]).sum() + 0.5 * (ref == v[i]).sum()) / len(ref)
        out.loc[g.index] = res
    return out


def llm_scores() -> pd.DataFrame:
    frames = []
    for k in ("main", "mainr"):
        p = LLM / f"replies_{k}.csv.gz"
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"accession": str}))
    if not frames:
        return pd.DataFrame(columns=["acc", "tone"])
    r = pd.concat(frames, ignore_index=True)
    r = r[r["parse_ok"] == True]                                   # noqa: E712
    r = r.drop_duplicates("accession", keep="last")              # retry wins
    return r.rename(columns={"accession": "acc"})[["acc", "tone", "materiality"]]


def filing_signals(panel_dir: Path) -> pd.DataFrame:
    idx = pd.read_csv(RES / "filings_index.csv.gz", dtype={"cik": str})
    idx = idx[idx["prior_accession"].fillna("") != ""].copy()
    idx["adate"] = accept_date(idx["accepted"])
    idx["acc"] = idx["accession"].str.replace("-", "", regex=False)
    sc = llm_scores()
    idx = idx.merge(sc, on="acc", how="left")
    st = idx[["rf_status", "legal_status", "mdna_status"]]
    any_found = st.isin(["changed", "nochange"]).any(axis=1)
    any_changed = (st == "changed").any(axis=1)
    idx.loc[any_found & ~any_changed, "tone"] = 0.0               # no call, tone 0
    idx.loc[~any_found, "tone"] = np.nan
    idx["F1"] = ref_percentile(idx, "f1_cos")
    idx["F2"] = ref_percentile(idx, "f2_cos")
    idx["F3"] = ref_percentile(idx, "tone")
    idx["F4"] = np.where(idx["F3"].isna(), np.nan, np.where(idx["F1"] <= F4_GATE, idx["F3"], 0.5))
    # Filing -> ticker by price life; ambiguous CIK/date pairs are dropped.
    life = pd.read_csv(panel_dir / "ticker_life.csv", dtype={"cik": str},
                       parse_dates=["firstpricedate", "lastpricedate"])
    life["cik"] = life["cik"].str.zfill(10)
    idx["cik"] = idx["cik"].str.zfill(10)
    m = idx.merge(life, on="cik", how="inner")
    m = m[(m["adate"] >= m["firstpricedate"]) & (m["adate"] <= m["lastpricedate"])]
    n = m.groupby("acc")["ticker"].transform("size")
    m = m[n == 1]
    return m[["ticker", "cik", "acc", "form", "adate", "f1_cos", "f2_cos", "tone",
              "F1", "F2", "F3", "F4"]].sort_values(["ticker", "adate"])


def monthly_signals(fs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, t): per signal, the most recent filing with that
    signal accepted on a date before t and at most 91 days before t."""
    base = panel[["ticker", "t"]].drop_duplicates().sort_values("t")
    out = base.copy()
    for s in SIGNALS:
        f = fs[fs[s].notna()][["ticker", "adate", s]].sort_values("adate").copy()
        f["key"] = f["adate"] + pd.Timedelta(days=1)             # strictly before t
        f["src"] = f["adate"]
        j = pd.merge_asof(base, f[["ticker", "key", "src", s]].sort_values("key"),
                          left_on="t", right_on="key", by="ticker", direction="backward")
        ok = (j["t"] - j["src"]).dt.days <= MAX_AGE_DAYS
        out[s] = np.where(ok, j[s], np.nan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolios
# ─────────────────────────────────────────────────────────────────────────────

class Data:
    def __init__(self, panel_dir: Path):
        p = pd.read_csv(panel_dir / "panel_monthly.csv.gz", parse_dates=["t"])
        sig = pd.read_csv(RES / "signals_monthly.csv.gz", parse_dates=["t"])
        p = p.merge(sig, on=["ticker", "t"], how="left")
        self.months = pd.DatetimeIndex(sorted(p["t"].unique()))
        self.tickers = np.array(sorted(p["ticker"].unique()))
        ti = {t: i for i, t in enumerate(self.tickers)}
        mi = {t: i for i, t in enumerate(self.months)}
        T, N = len(self.months), len(self.tickers)
        r, c = p["t"].map(mi).values, p["ticker"].map(ti).values
        def mat(col, fill=np.nan, dtype=float):
            M = np.full((T, N), fill, dtype=dtype)
            M[r, c] = p[col].values
            return M
        self.ret = mat("ret_next")
        self.member = mat("member", False, bool)
        self.price = mat("close_unadj")
        self.mcap = mat("mcap")
        self.bm = mat("bm")
        self.mom = mat("mom12_1")
        self.rev = mat("rev1")
        sectors = sorted(p["sector"].dropna().unique())
        self.sector_names = sectors
        p["sec_code"] = p["sector"].map({s: i for i, s in enumerate(sectors)}).fillna(-1)
        self.sector = mat("sec_code", -1, float).astype(int)
        self.sig = {s: mat(s) for s in SIGNALS}
        self.base_ok = self.member & (self.price >= MIN_PRICE) & np.isfinite(self.ret)
        self.ret0 = np.nan_to_num(self.ret, nan=0.0)              # held but no price -> cash
        self.month_label = self.months.to_period("M")

    def ret_month_index(self, a: str, b: str) -> np.ndarray:
        """Formation rows t whose return month (t+1) lies in [a, b]."""
        rm = (self.month_label + 1)
        return np.where((rm >= pd.Period(a, "M")) & (rm <= pd.Period(b, "M")))[0]


def groups_for(sec_row: np.ndarray, ok: np.ndarray) -> np.ndarray:
    g = np.where(ok, sec_row, -2)
    out = g.copy()
    for s in np.unique(g[ok]):
        if (g == s).sum() < MIN_SECTOR:
            out[g == s] = 99                                     # "other"
    return out


def within_pct(vals: np.ndarray, grp: np.ndarray, ok: np.ndarray) -> np.ndarray:
    pct = np.full(len(vals), np.nan)
    for s in np.unique(grp[ok]):
        m = ok & (grp == s)
        v = vals[m]
        order = np.argsort(v, kind="mergesort")
        sv = v[order]
        less = np.searchsorted(sv, v, side="left")
        eq = np.searchsorted(sv, v, side="right") - less
        pct[np.where(m)[0]] = (less + 0.5 * eq) / len(v)
    return pct


def formation_weights(D: Data, sig: np.ndarray, rows: np.ndarray, vw: bool = False,
                      rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    T, N = sig.shape
    W = np.zeros((T, N))
    PCT = np.full((T, N), np.nan)
    for t in rows:
        ok = D.base_ok[t] & np.isfinite(sig[t]) & (D.sector[t] >= 0)
        if ok.sum() < 50:
            continue
        grp = groups_for(D.sector[t], ok)
        vals = sig[t].copy()
        if rng is not None:                                      # placebo: shuffle within sector
            for s in np.unique(grp[ok]):
                ii = np.where(ok & (grp == s))[0]
                vals[ii] = vals[rng.permutation(ii)]
        pct = within_pct(vals, grp, ok)
        PCT[t] = pct
        lg = ok & (pct > 1 - QUANT)
        sh = ok & (pct < QUANT)
        if lg.sum() == 0 or sh.sum() == 0:
            continue
        if vw:
            mc = np.nan_to_num(D.mcap[t], nan=0.0)
            W[t, lg] = mc[lg] / mc[lg].sum()
            W[t, sh] = -mc[sh] / mc[sh].sum()
        else:
            W[t, lg] = 1.0 / lg.sum()
            W[t, sh] = -1.0 / sh.sum()
    return W, PCT


def strategy_returns(D: Data, W: np.ndarray, K: int, rows: np.ndarray) -> pd.DataFrame:
    """Return month t+1 for formation rows t. K-month Jegadeesh-Titman average."""
    T = W.shape[0]
    WK = np.zeros_like(W)
    for k in range(K):
        WK[k:] += W[:T - k] if k else W
    WK /= K
    gross = (WK * D.ret0).sum(axis=1)
    turn = np.abs(np.diff(WK, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
    cost = COST_SIDE * turn
    borrow = BORROW_M * np.abs(np.minimum(WK, 0)).sum(axis=1)
    net = gross - cost - borrow
    rm = (D.month_label + 1).astype(str)
    df = pd.DataFrame({"month": rm, "gross": gross, "cost": cost, "borrow": borrow,
                       "net": net, "turnover": turn,
                       "n_long": (WK > 0).sum(axis=1), "n_short": (WK < 0).sum(axis=1)})
    return df.iloc[rows].reset_index(drop=True), WK


def all_rows_for(D: Data, a: str, b: str, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Formation rows needed (including K-1 earlier ones) and the evaluated rows."""
    ev = D.ret_month_index(a, b)
    # K earlier formations too, so the first month's turnover is measured
    # against a fully formed previous book (not against zero).
    need = np.arange(max(0, ev[0] - K), ev[-1] + 1)
    return need, ev


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def nw_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / T
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / T
    return float(x.mean() / math.sqrt(s / T)) if s > 0 else float("nan")


def sharpe_a(x) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(12)) if sd > 0 else float("nan")


def stationary_idx(T: int, block: int, rng) -> np.ndarray:
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(T)
    for i in range(1, T):
        idx[i] = rng.integers(T) if rng.random() < 1.0 / block else (idx[i - 1] + 1) % T
    return idx


def reality_check(R: np.ndarray, B: int, block: int, rng) -> dict:
    """White's Reality Check, studentized, stationary bootstrap. R: (T, K)."""
    T = R.shape[0]
    mu, sd = R.mean(0), R.std(0, ddof=1)
    V = float(np.max(math.sqrt(T) * mu / sd))
    Vs = np.empty(B)
    for b in range(B):
        i = stationary_idx(T, block, rng)
        Vs[b] = np.max(math.sqrt(T) * (R[i].mean(0) - mu) / sd)
    return {"statistic": V, "p_value": float((1 + (Vs >= V).sum()) / (B + 1)), "B": B,
            "mean_block_months": block, "settings": int(R.shape[1])}


def dsr(r: np.ndarray, sr_all: np.ndarray, N: int) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), monthly units."""
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    v = float(np.var(np.asarray(sr_all) / math.sqrt(12), ddof=1))
    sr0 = math.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    return {"dsr": float(norm.cdf(z)), "sharpe_annual": float(sr * math.sqrt(12)),
            "hurdle_sharpe_annual": float(sr0 * math.sqrt(12)), "trials": N,
            "skew": g3, "kurtosis": g4}


def summary(df: pd.DataFrame) -> dict:
    x = df["net"].values
    out = {"months": int(len(x)), "net_mean_pct": float(x.mean() * 100),
           "net_t_nw": nw_t(x), "net_sharpe": sharpe_a(x),
           "gross_mean_pct": float(df["gross"].mean() * 100), "gross_t_nw": nw_t(df["gross"].values),
           "gross_sharpe": sharpe_a(df["gross"].values),
           "cost_pct_per_month": float((df["cost"] + df["borrow"]).mean() * 100),
           "turnover_per_month": float(df["turnover"].mean()),
           "avg_names_long": float(df["n_long"].mean()), "avg_names_short": float(df["n_short"].mean())}
    yr = df.assign(y=df["month"].str[:4]).groupby("y")["net"].apply(lambda s: (1 + s).prod() - 1)
    out["net_by_year_pct"] = {k: round(float(v) * 100, 2) for k, v in yr.items()}
    return out


def fama_macbeth(D: Data, PCT: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """Next-month return on the within-sector signal percentile (centered),
    sector dummies and percentile-ranked controls (median-filled)."""
    out = []
    for t in rows:
        ok = D.base_ok[t] & np.isfinite(PCT[t])
        if ok.sum() < 50:
            continue
        y = D.ret[t, ok]
        s = PCT[t, ok] - 0.5
        X = [np.ones(ok.sum()), s]
        for M in (np.log(D.mcap[t]), np.log(D.bm[t]), D.mom[t], D.rev[t]):
            v = np.where(np.isfinite(M), M, np.nan)[ok]
            r = pd.Series(v).rank(pct=True).values
            X.append(np.where(np.isfinite(r), r, np.nanmedian(r)) - 0.5)
        sec = D.sector[t, ok]
        for c in np.unique(sec)[1:]:
            X.append((sec == c).astype(float))
        X = np.column_stack(X)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.append({"month": str(D.month_label[t] + 1), "slope": beta[1], "n": int(ok.sum())})
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Stages
# ─────────────────────────────────────────────────────────────────────────────

def setting_name(s: str, K: int) -> str:
    return f"{s}_K{K}"


def stage_signals(panel_dir: Path) -> None:
    fs = filing_signals(panel_dir)
    panel = pd.read_csv(panel_dir / "panel_monthly.csv.gz", parse_dates=["t"],
                        usecols=["ticker", "t", "member"])
    ms = monthly_signals(fs, panel)
    ms.to_csv(RES / "signals_monthly.csv.gz", index=False, float_format="%.6g")
    fs.to_csv(RES / "signals_filings.csv.gz", index=False, float_format="%.6g")
    mem = ms.merge(panel, on=["ticker", "t"])
    mem = mem[mem["member"]]
    cov = mem.groupby(mem["t"].dt.year)[SIGNALS].apply(lambda g: g.notna().mean()).round(3)
    stats = {"filings_with_signal": {s: int(fs[s].notna().sum()) for s in SIGNALS},
             "member_coverage_by_year": cov.to_dict(orient="index"),
             "tone_counts": fs["tone"].value_counts(dropna=False).sort_index().to_dict()}
    stats["member_coverage_by_year"] = {int(k): v for k, v in stats["member_coverage_by_year"].items()}
    stats["tone_counts"] = {str(k): int(v) for k, v in stats["tone_counts"].items()}
    (RES / "signal_coverage.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def run_settings(D: Data, a: str, b: str, rng=None, vw=False) -> dict:
    res = {}
    for s in SIGNALS:
        need3, _ = all_rows_for(D, a, b, 3)
        W, PCT = formation_weights(D, D.sig[s], need3, vw=vw, rng=rng)
        for K in HOLDS:
            _, ev = all_rows_for(D, a, b, K)
            df, _ = strategy_returns(D, W, K, ev)
            res[setting_name(s, K)] = (df, W, PCT)
    return res


def stage_search(panel_dir: Path) -> None:
    D = Data(panel_dir)
    a, b = SEARCH
    res = run_settings(D, a, b)
    names = list(res)
    rows, monthly = [], {}
    for n in names:
        df = res[n][0]
        sm = summary(df)
        h1 = df[df["month"] <= "2012-12"]["net"]
        h2 = df[df["month"] > "2012-12"]["net"]
        rows.append({"setting": n, **{k: v for k, v in sm.items() if k != "net_by_year_pct"},
                     "net_mean_2007_12_pct": float(h1.mean() * 100),
                     "net_mean_2013_18_pct": float(h2.mean() * 100)})
        monthly[n] = df.set_index("month")["net"]
    tab = pd.DataFrame(rows)
    order = tab.sort_values(["net_sharpe", "turnover_per_month"], ascending=[False, True])
    sel = order.iloc[0]["setting"]
    R = np.column_stack([monthly[n].values for n in names])
    rng = np.random.default_rng(SEED)
    rc = reality_check(R, RC_B, RC_BLOCK, rng)
    d = dsr(monthly[sel].values, tab["net_sharpe"].values, len(names))
    # Placebo: best-of-8 Sharpe with signals shuffled within sector each month
    best = []
    prng = np.random.default_rng(SEED + 1)
    for i in range(N_PLACEBO):
        pr = run_settings(D, a, b, rng=prng)
        best.append(max(sharpe_a(pr[n][0]["net"].values) for n in pr))
        if (i + 1) % 50 == 0:
            print(f"[search] placebo {i + 1}/{N_PLACEBO}", flush=True)
    best = np.array(best)
    sel_sh = float(tab.set_index("setting").loc[sel, "net_sharpe"])
    vw = run_settings(D, a, b, vw=True)
    fm = fama_macbeth(D, res[sel][2], D.ret_month_index(a, b))
    out = {"selected": sel, "rule": "highest net Sharpe on return months 2007-02..2018-12; ties lower turnover",
           "selected_summary": summary(res[sel][0]),
           "reality_check": rc, "deflated_sharpe": d,
           "placebo": {"runs": N_PLACEBO, "share_best_of_8_at_least_selected": float((best >= sel_sh).mean()),
                       "best_of_8_quantiles": {str(q): float(np.quantile(best, q)) for q in (.5, .9, .95, .99)}},
           "fama_macbeth_selected_search": {"mean_slope_pct": float(fm["slope"].mean() * 100),
                                            "t_nw": nw_t(fm["slope"].values), "months": int(len(fm))},
           "value_weighted_net_sharpe": {n: sharpe_a(vw[n][0]["net"].values) for n in vw}}
    tab.to_csv(RES / "search_results.csv", index=False, float_format="%.5g")
    pd.DataFrame(monthly).to_csv(RES / "search_monthly_net.csv", float_format="%.6g")
    (RES / "selection.json").write_text(json.dumps(out, indent=1))
    print(tab.to_string())
    print(json.dumps(out, indent=1))


def _selection_committed() -> dict:
    p = RES / "selection.json"
    if not p.exists():
        raise SystemExit("no selection.json: run the search stage and commit it first")
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "--", str(p)],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(p)],
                           capture_output=True, text=True).stdout.strip()
    if not log or dirty:
        raise SystemExit("selection.json must be committed and unchanged before the holdout")
    return {**json.loads(p.read_text()), "commit": log.splitlines()[0]}


def stage_holdout(panel_dir: Path) -> None:
    sel = _selection_committed()
    D = Data(panel_dir)
    a, b = HOLDOUT
    res = run_settings(D, a, b)
    name = sel["selected"]
    df, W, PCT = res[name]
    K = int(name.split("_K")[1])
    _, WK = strategy_returns(D, W, K, D.ret_month_index(a, b))
    x = df["net"].values
    halves = [df[(df["month"] >= h[0]) & (df["month"] <= h[1])]["net"] for h in HALVES]
    k3 = int(math.ceil(0.03 * len(x)))
    trimmed = np.sort(x)[:-k3]
    ev = D.ret_month_index(a, b)
    contrib = (WK[ev] * D.ret0[ev]).sum(axis=0)
    total_net = float(x.sum())
    top = int(np.argmax(contrib))
    top_share = float(contrib[top] / total_net) if total_net > 0 else float("nan")
    strat = {"t_nw": nw_t(x), "half_means_pct": [float(h.mean() * 100) for h in halves],
             "mean_without_best_3pct_pct": float(trimmed.mean() * 100), "best_months_removed": k3,
             "top_stock_share_of_net_profit": top_share}
    strat["pass"] = bool(strat["t_nw"] >= 2.0 and all(h.mean() > 0 for h in halves)
                         and trimmed.mean() > 0 and total_net > 0 and top_share <= 0.25)
    fm = fama_macbeth(D, PCT, ev)
    fh = [fm[(fm["month"] >= h[0]) & (fm["month"] <= h[1])]["slope"] for h in HALVES]
    sig = {"mean_slope_pct": float(fm["slope"].mean() * 100), "t_nw": nw_t(fm["slope"].values),
           "half_means_pct": [float(h.mean() * 100) for h in fh], "months": int(len(fm))}
    sig["pass"] = bool(sig["t_nw"] >= 2.0 and all(h.mean() > 0 for h in fh))
    outcome = "PASS" if strat["pass"] else ("SIGNAL ONLY" if sig["pass"] else "FAIL")
    allset = []
    for n, (d2, _, P2) in res.items():
        f2 = fama_macbeth(D, P2, ev)
        allset.append({"setting": n, **{k: v for k, v in summary(d2).items() if k != "net_by_year_pct"},
                       "fm_t_nw": nw_t(f2["slope"].values)})
    vw = run_settings(D, a, b, vw=True)
    out = {"selected": name, "selection_commit": sel["commit"], "outcome": outcome,
           "strategy_verdict": strat, "signal_verdict": sig, "summary": summary(df),
           "value_weighted_selected": summary(vw[name][0])}
    (RES / "holdout.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(allset).to_csv(RES / "holdout_all_settings.csv", index=False, float_format="%.5g")
    pd.DataFrame({n: r[0].set_index("month")["net"] for n, r in res.items()}).to_csv(
        RES / "holdout_monthly_net.csv", float_format="%.6g")
    pd.DataFrame({"ticker": D.tickers, "contribution": contrib}).sort_values(
        "contribution", ascending=False).to_csv(panel_dir / "holdout_contrib_PRIVATE.csv", index=False)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["signals", "search", "holdout"])
    ap.add_argument("--panel", required=True)
    a = ap.parse_args()
    {"signals": stage_signals, "search": stage_search, "holdout": stage_holdout}[a.stage](Path(a.panel))


if __name__ == "__main__":
    main()
