"""
Analysis of results/analysis/calm_test.csv: do calmer, lower-error stocks give
bigger option peaks?

Market regime moves every option in a week together (April 2025 lifted every
put), so the main tests compare names within the same week: each measure is
ranked inside its week, and the week-by-week correlations are counted (how
many of the 52 weeks point the same way), which avoids treating ~2,000
overlapping options as independent.

    python -m replay.calm_analysis      # prints, and writes results/analysis/calm_summary.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

from . import config

R = config.RESULTS_DIR / "analysis"
MEASURES = {"avg_mae_pct": "LSTM error (MAE %)", "hit_rate_pct": "LSTM within ±2%",
            "vol_1y_pct": "1-year volatility", "vol_20d_pct": "20-day volatility",
            "beta_1y": "Stock beta", "iv_pct": "Implied vol at entry"}
OUTCOMES = {"peak_pct": "Peak (best bid)", "peak5_pct": "Peak within 5 days",
            "end_pct": "Held to Thursday exit"}


def load() -> pd.DataFrame:
    d = pd.read_csv(R / "calm_test.csv")
    d = d[d["peak_pct"].notna()].copy()
    d["tradable"] = (d["spread_pct"] <= 20) & (d["delta"].abs() >= 0.35)
    d["hit50"] = d["peak_pct"] >= 50
    d["hit100"] = d["peak_pct"] >= 100
    return d


def within_week(d: pd.DataFrame, col: str) -> pd.Series:
    return d.groupby("asof")[col].rank(pct=True)


def weekly_rhos(d: pd.DataFrame, x: str, y: str) -> pd.Series:
    out = {}
    for a, g in d.groupby("asof"):
        g = g[[x, y]].dropna()
        if len(g) >= 8 and g[x].nunique() > 2:
            out[a] = spearmanr(g[x], g[y])[0]
    return pd.Series(out)


def quintiles(d: pd.DataFrame, by: str) -> pd.DataFrame:
    q = pd.qcut(within_week(d, by), 5, labels=["1 calmest", "2", "3", "4", "5 most volatile"]
                if by != "hit_rate_pct" else ["1 lowest", "2", "3", "4", "5 highest"])
    g = d.groupby(q, observed=True)
    return pd.DataFrame({
        "options": g.size(),
        f"median {by}": g[by].median().round(2),
        "median IV %": g["iv_pct"].median().round(1),
        "median leverage": g["leverage"].apply(lambda s: s.abs().median()).round(1),
        "median peak %": g["peak_pct"].median().round(1),
        "peak >= +50%": (g["hit50"].mean() * 100).round(1),
        "peak >= +100%": (g["hit100"].mean() * 100).round(1),
        "median at exit %": g["end_pct"].median().round(1),
        "mean at exit %": g["end_pct"].mean().round(1),
        "median biggest stock move %": g.apply(lambda x: np.where(x["side"] == "CALL", x["max_up_pct"],
                                                                  x["max_down_pct"]).mean(),
                                               include_groups=False).round(2),
    })


def summary(d: pd.DataFrame, label: str) -> dict:
    res = {"label": label, "options": int(len(d)), "weeks": int(d["asof"].nunique()),
           "tickers": int(d["ticker"].nunique()), "tests": {}}
    for x, xn in MEASURES.items():
        for y, yn in OUTCOMES.items():
            sub = d[[x, y, "asof"]].dropna()
            if len(sub) < 20:
                continue
            pooled = spearmanr(sub[x], sub[y])[0]
            wr = weekly_rhos(sub, x, y)
            neg = int((wr < 0).sum()); pos = int((wr > 0).sum())
            res["tests"][f"{x}|{y}"] = {
                "measure": xn, "outcome": yn, "pooled_rho": round(float(pooled), 3),
                "median_weekly_rho": round(float(wr.median()), 3) if len(wr) else None,
                "weeks_negative": neg, "weeks_positive": pos,
                "sign_test_p": round(float(binomtest(neg, neg + pos).pvalue), 4) if neg + pos else None,
                "wilcoxon_p": round(float(wilcoxon(wr).pvalue), 4) if len(wr) >= 10 else None,
            }
    return res


def partial_mae_given_iv(d: pd.DataFrame) -> dict:
    """Does LSTM error still matter once you know the option's implied vol?
    Within-week ranks, residualise peak and MAE on IV, then correlate."""
    x = d[["asof", "avg_mae_pct", "iv_pct", "peak_pct"]].dropna().copy()
    for c in ["avg_mae_pct", "iv_pct", "peak_pct"]:
        x[c] = within_week(x, c)
    def resid(y):
        b = np.polyfit(x["iv_pct"], x[y], 1)
        return x[y] - np.polyval(b, x["iv_pct"])
    r, p = spearmanr(resid("avg_mae_pct"), resid("peak_pct"))
    return {"partial_rho_mae_peak_given_iv": round(float(r), 3), "p_naive": round(float(p), 4), "n": int(len(x))}


def main():
    d = load()
    out = {"all": summary(d, "all options"),
           "tradable": summary(d[d["tradable"]], "agent-tradable (spread <= 20%, delta >= 0.35)"),
           "candidates": summary(d[d["group"] == "candidate"], "stage-1 candidates"),
           "controls": summary(d[d["group"] == "control"], "random S&P 500 controls"),
           "calls": summary(d[d["side"] == "CALL"], "calls"),
           "puts": summary(d[d["side"] == "PUT"], "puts"),
           "partial": partial_mae_given_iv(d)}
    tabs = {}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for by in ["avg_mae_pct", "vol_1y_pct", "iv_pct", "hit_rate_pct"]:
        t = quintiles(d, by)
        tabs[by] = json.loads(t.reset_index().to_json(orient="records"))
        print(f"\n== quintiles by {by} (ranked within each week) ==\n{t.to_string()}")
    out["quintiles"] = tabs
    out["quintiles_tradable_mae"] = json.loads(quintiles(d[d["tradable"]], "avg_mae_pct").reset_index()
                                               .to_json(orient="records"))
    print("\n== tradable only, by MAE ==\n" + quintiles(d[d["tradable"]], "avg_mae_pct").to_string())
    for k in ["all", "tradable", "candidates", "controls", "calls", "puts"]:
        print(f"\n== {out[k]['label']}: {out[k]['options']} options, {out[k]['tickers']} tickers, "
              f"{out[k]['weeks']} weeks ==")
        if not out[k]["tests"]:
            continue
        print(pd.DataFrame(out[k]["tests"]).T[["measure", "outcome", "pooled_rho", "median_weekly_rho",
                                                 "weeks_negative", "weeks_positive", "sign_test_p",
                                                 "wilcoxon_p"]].to_string())
    print("\n", out["partial"])
    (R / "calm_summary.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
