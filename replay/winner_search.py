"""
Is there anything, knowable at 9:46 on the entry morning, that picks out the
winners or weeds out the losers?  Searched on the 1,518 real-option trades of
the calm-stock test (results/analysis/calm_features.csv).

Outcomes
  win50   the bid reached +50% over the price paid at some point (a +50% limit
          sell would have filled): a "potential winner"
  dud     the bid never got 10% above the price paid: no chance to exit ahead
  tp50    what a +50% take-profit then hold-to-Thursday rule actually returns
          (+50% if win50, else the Thursday-exit value)

Guarding against luck (dozens of features are tried):
  * every feature and outcome is ranked inside its own week, so a market-wide
    week (April 2025) cannot create a pattern;
  * the year is split in half: a pattern counts only if it shows up, the same
    way round, in January-June AND in July-December;
  * models are fitted on one half and scored on the other half they never saw.

    python -m replay.winner_search      # prints; writes results/analysis/winner_search.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

R = config.RESULTS_DIR / "analysis"
SPLIT = "2025-07-01"

FEATURES = {
    # LSTM / funnel
    "avg_mae_pct": "LSTM error (MAE)", "hit_rate_pct": "LSTM within ±2%", "dir_accuracy_pct": "LSTM direction accuracy",
    "avg_snr": "LSTM signal/noise", "is_candidate": "Was a stage-1 candidate",
    # stock character
    "vol_1y_pct": "1-year volatility", "vol_20d_pct": "20-day volatility", "beta_1y": "Stock beta",
    "idio_vol_1y_pct": "Stock-specific volatility", "atr14_pct": "Average daily range (ATR)",
    "prev_range_pct": "Yesterday's range", "vol_ratio_prev": "Yesterday's volume vs normal",
    "stock_price": "Stock price",
    # earnings
    "earnings_in_window": "Earnings before the Thursday exit", "days_to_earnings": "Days to next earnings",
    "days_since_earnings": "Days since last earnings",
    # option
    "is_call": "Call (vs put)", "iv_pct": "Implied vol", "iv_over_rv20": "Implied vol / 20-day realised vol",
    "spread_pct": "Bid-ask spread at entry", "dte": "Days to expiry", "abs_delta": "Delta",
    "abs_leverage": "Leverage", "premium_dollars": "Contract cost ($)", "premium_pct_of_stock": "Premium as % of stock",
    "otm_pct": "How far out of the money",
    # direction-bearing, signed: + = points the option's way
    "gap_pct_with": "Opening gap, the option's way", "early_move_pct_with": "Open-to-9:46 move, the option's way",
    "ret_1d_pct_with": "Yesterday's move, the option's way", "ret_5d_pct_with": "5-day move, the option's way",
    "ret_20d_pct_with": "20-day move, the option's way", "ret_60d_pct_with": "60-day move, the option's way",
    "dist_ma20_pct_with": "Distance from 20-day average, the option's way",
    "dist_ma50_pct_with": "Distance from 50-day average, the option's way",
    "rsi_minus_50_with": "RSI, the option's way", "dist_52w_high_pct_with": "Distance from 52-week high, the option's way",
    "spy_vs_ma20_pct_with": "SPY vs its 20-day average, the option's way",
    "spy_ret_5d_pct_with": "SPY 5-day move, the option's way", "spy_gap_pct_with": "SPY opening gap, the option's way",
}
OUTCOMES = {"win50": "Reached +50%", "dud": "Never reached +10%", "tp50": "+50% take-profit rule return"}


def load() -> pd.DataFrame:
    d = pd.read_csv(R / "calm_features.csv")
    d["half"] = np.where(d["asof"] < SPLIT, "H1", "H2")
    d["win50"] = (d["peak_pct"] >= 50).astype(int)
    d["dud"] = (d["peak_pct"] < 10).astype(int)
    d["tp50"] = np.where(d["peak_pct"] >= 50, 50.0, d["end_pct"])
    d["is_candidate"] = (d["group"] == "candidate").astype(int)
    d["is_call"] = (d["side"] == "CALL").astype(int)
    d["abs_delta"] = d["delta"].abs()
    d["abs_leverage"] = d["leverage"].abs()
    return d


def wrank(d: pd.DataFrame, cols) -> pd.DataFrame:
    """Rank inside each week (0..1); binary features stay as they are."""
    out = pd.DataFrame(index=d.index)
    for c in cols:
        if d[c].dropna().isin([0, 1]).all():
            out[c] = d[c]
        else:
            out[c] = d.groupby("asof")[c].rank(pct=True)
    return out


def screen(d: pd.DataFrame) -> pd.DataFrame:
    X = wrank(d, list(FEATURES))
    rows = []
    for f, name in FEATURES.items():
        for o in OUTCOMES:
            r = {"feature": f, "name": name, "outcome": o}
            for h in ("H1", "H2"):
                m = (d["half"] == h) & X[f].notna()
                y = d.loc[m, o] if o != "tp50" else d.loc[m].groupby("asof")[o].rank(pct=True)
                rho, p = spearmanr(X.loc[m, f], y)
                r[f"rho_{h}"], r[f"p_{h}"] = rho, p
            r["same_sign"] = np.sign(r["rho_H1"]) == np.sign(r["rho_H2"])
            r["replicates"] = bool(r["same_sign"] and r["p_H1"] < 0.01 and r["p_H2"] < 0.01)
            rows.append(r)
    return pd.DataFrame(rows)


def quintile_table(d: pd.DataFrame, f: str) -> pd.DataFrame:
    if d[f].dropna().isin([0, 1]).all():
        g = d.groupby(d[f].map({0: "no", 1: "yes"}))
    else:
        g = d.groupby(pd.qcut(d.groupby("asof")[f].rank(pct=True), 5,
                              labels=["lowest", "2", "3", "4", "highest"]), observed=True)
    out = pd.DataFrame({"options": g.size(),
                        "reached +50%": (g["win50"].mean() * 100).round(1),
                        "never +10%": (g["dud"].mean() * 100).round(1),
                        "TP-50 rule avg %": g["tp50"].mean().round(1),
                        "H1 reached +50%": g.apply(lambda x: x.loc[x["half"] == "H1", "win50"].mean() * 100,
                                                   include_groups=False).round(1),
                        "H2 reached +50%": g.apply(lambda x: x.loc[x["half"] == "H2", "win50"].mean() * 100,
                                                   include_groups=False).round(1),
                        "H1 TP-50 avg %": g.apply(lambda x: x.loc[x["half"] == "H1", "tp50"].mean(),
                                                  include_groups=False).round(1),
                        "H2 TP-50 avg %": g.apply(lambda x: x.loc[x["half"] == "H2", "tp50"].mean(),
                                                  include_groups=False).round(1)})
    return out


def models(d: pd.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    cols = list(FEATURES)
    X = wrank(d, cols).fillna(0.5)
    res = {}
    for o in ("win50", "dud"):
        for train, test in (("H1", "H2"), ("H2", "H1")):
            tr, te = d["half"] == train, d["half"] == test
            for name, m in (("logistic", LogisticRegression(C=0.3, max_iter=2000)),
                            ("boosted trees", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                             max_iter=150, random_state=0))):
                m.fit(X[tr], d.loc[tr, o])
                p = m.predict_proba(X[te])[:, 1]
                auc = roc_auc_score(d.loc[te, o], p)
                q = pd.qcut(pd.Series(p, index=d.index[te]).rank(method="first"), 5, labels=False)
                top, bot = q == 4, q == 0
                key = f"{o}|{name}|train {train} -> test {test}"
                res[key] = {"auc": round(float(auc), 3),
                            "test win50 top fifth": round(float(d.loc[te][top.values]["win50"].mean() * 100), 1),
                            "test win50 bottom fifth": round(float(d.loc[te][bot.values]["win50"].mean() * 100), 1),
                            "test tp50 top fifth": round(float(d.loc[te][top.values]["tp50"].mean()), 1),
                            "test tp50 bottom fifth": round(float(d.loc[te][bot.values]["tp50"].mean()), 1),
                            "test base win50": round(float(d.loc[te, "win50"].mean() * 100), 1),
                            "test base tp50": round(float(d.loc[te, "tp50"].mean()), 1)}
    return res


def direction_rules(d: pd.DataFrame) -> pd.DataFrame:
    """Pick ONE side per stock-week by a rule; compare with the side it rejects."""
    pairs = d.pivot_table(index=["asof", "ticker"], columns="side",
                          values=["win50", "tp50", "half"], aggfunc="first").dropna()
    feat = d[d["side"] == "CALL"].set_index(["asof", "ticker"])
    rules = {"Follow SPY trend (above 20-day avg -> call)": feat["spy_vs_ma20_pct"],
             "Follow stock's 20-day trend": feat["dist_ma20_pct"],
             "Follow stock's 5-day move": feat["ret_5d_pct"],
             "Fade stock's 5-day move": -feat["ret_5d_pct"],
             "Follow the opening gap": feat["gap_pct"],
             "Fade the opening gap": -feat["gap_pct"],
             "Follow open-to-9:46 move": feat["early_move_pct"],
             "Fade open-to-9:46 move": -feat["early_move_pct"],
             "RSI: oversold -> call, overbought -> put": -feat["rsi_minus_50"],
             "Always call": pd.Series(1.0, index=feat.index)}
    rows = []
    for name, s in rules.items():
        s = s.reindex(pairs.index)
        ok = s.notna() & (s != 0)
        pick_call = (s > 0)[ok]
        p = pairs[ok]
        chosen_w = np.where(pick_call, p[("win50", "CALL")], p[("win50", "PUT")]).astype(float)
        other_w = np.where(pick_call, p[("win50", "PUT")], p[("win50", "CALL")]).astype(float)
        chosen_t = np.where(pick_call, p[("tp50", "CALL")], p[("tp50", "PUT")]).astype(float)
        other_t = np.where(pick_call, p[("tp50", "PUT")], p[("tp50", "CALL")]).astype(float)
        half = p[("half", "CALL")].values
        r = {"rule": name, "stock-weeks": int(ok.sum()),
             "chosen reached +50%": round(chosen_w.mean() * 100, 1),
             "rejected reached +50%": round(other_w.mean() * 100, 1),
             "chosen TP-50 avg %": round(chosen_t.mean(), 1), "rejected TP-50 avg %": round(other_t.mean(), 1)}
        for h in ("H1", "H2"):
            m = half == h
            r[f"{h} edge TP-50 pts"] = round(chosen_t[m].mean() - other_t[m].mean(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    d = load()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    base = {"options": len(d), "win50": round(d["win50"].mean() * 100, 1), "dud": round(d["dud"].mean() * 100, 1),
            "tp50_mean": round(d["tp50"].mean(), 1), "end_mean": round(d["end_pct"].mean(), 1),
            "H1": int((d["half"] == "H1").sum()), "H2": int((d["half"] == "H2").sum())}
    print("BASE", base)
    sc = screen(d)
    print("\n== screen (within-week ranks; replicates = same sign and p<0.01 in both halves) ==")
    print(sc.sort_values(["outcome", "rho_H1"]).round(3).to_string())
    rep = sc[sc["replicates"]]
    qt = {}
    for f in sorted(set(rep["feature"])):
        t = quintile_table(d, f)
        qt[f] = json.loads(t.reset_index().astype({t.index.name or "index": str} if False else {}).to_json(orient="records"))
        print(f"\n== {FEATURES[f]} ==\n{t.to_string()}")
    mo = models(d)
    print("\n== models (fit on one half, scored on the other) ==")
    print(pd.DataFrame(mo).T.to_string())
    dr = direction_rules(d)
    print("\n== direction rules ==\n" + dr.to_string())
    out = {"base": base, "screen": json.loads(sc.to_json(orient="records")), "quintiles": qt,
           "models": mo, "direction_rules": json.loads(dr.to_json(orient="records"))}
    (R / "winner_search.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
