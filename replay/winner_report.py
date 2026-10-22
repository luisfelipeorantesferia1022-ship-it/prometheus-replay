"""
Tables for the winner/loser search (results/analysis/winner_search.json and
calm_features.csv), as one HTML page.

    python -m replay.winner_report      # results/report/winner_search_tables.html
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from . import config
from .report_tables import CSS, table
from .winner_search import FEATURES, load

R = config.RESULTS_DIR


def p(x, d=0, sign=False):
    return "n/a" if pd.isna(x) else (f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%")


def main():
    d = load()
    s = json.loads((R / "analysis" / "winner_search.json").read_text())
    sections = []
    b = s["base"]
    sections.append(("The question and the data",
                     "Can anything known at 9:46 on the entry morning pick out the winners, or at least weed out "
                     "the losers? Searched on the 1,518 real-option trades of the calm-stock test (both the call "
                     "and the put for each stock-week, bought at the ask at 9:46, followed to the Thursday exit). "
                     "A pattern only counts if it shows up the same way in January-June and in July-December.",
                     table(["", "Value"], [
                         ["Options", f"{b['options']:,} ({b['H1']} in Jan-Jun, {b['H2']} in Jul-Dec)"],
                         ["Reached +50% at some point (potential winners)", p(b["win50"], 1)],
                         ["Never got 10% above the price paid (duds)", p(b["dud"], 1)],
                         ["Average return with a +50% take-profit, else hold to Thursday", p(b["tp50_mean"], 1, True)],
                         ["Features tested", str(len(FEATURES))]])))

    # spread buckets
    bins = [0, 5, 10, 15, 20, 30, 1e9]
    labels = ["under 5%", "5-10%", "10-15%", "15-20%", "20-30%", "over 30%"]
    g = d.groupby(pd.cut(d["spread_pct"], bins, labels=labels, right=False), observed=True)
    rows = []
    for k, x in g:
        h1, h2 = x[x["half"] == "H1"], x[x["half"] == "H2"]
        rows.append([k, len(x), p(x["win50"].mean() * 100), p(x["dud"].mean() * 100),
                     f"<b>{p(x['tp50'].mean(), 1, True)}</b>", p(h1["tp50"].mean(), 1, True),
                     p(h2["tp50"].mean(), 1, True)])
    sections.append(("The one thing that held up: the bid-ask spread at entry",
                     "Spread = (ask - bid) / mid at 9:46. The agent already skips spreads over 20%. Tighter is "
                     "better in both halves of the year. Under 5% spread these were mostly mega-caps (GOOGL, BAC, "
                     "GS, NVDA, META, MSFT); only 86 options, so the break-even there is uncertain (90% range "
                     "roughly -10% to +12%).",
                     table(["Spread at entry", "Options", "Reached +50%", "Duds (never +10%)",
                            "Take-profit-50 avg", "Jan-Jun", "Jul-Dec"], rows)))

    # screen summary
    sc = pd.DataFrame(s["screen"])
    keep = ["spread_pct", "avg_mae_pct", "hit_rate_pct", "dir_accuracy_pct", "is_candidate", "vol_1y_pct",
            "beta_1y", "iv_pct", "iv_over_rv20", "earnings_in_window", "premium_dollars", "abs_leverage", "dte",
            "is_call", "gap_pct_with", "ret_20d_pct_with", "rsi_minus_50_with", "spy_vs_ma20_pct_with",
            "early_move_pct_with", "vol_ratio_prev"]
    rows = []
    for f in keep:
        a = sc[(sc["feature"] == f) & (sc["outcome"] == "win50")].iloc[0]
        t = sc[(sc["feature"] == f) & (sc["outcome"] == "tp50")].iloc[0]
        if a["replicates"] or t["replicates"]:
            r = sc[(sc["feature"] == f) & sc["replicates"]]
            strong = (r[["rho_H1", "rho_H2"]].abs().min(axis=1) >= 0.12).any()
            verdict = "<b>Holds in both halves</b>" if strong else "Weak, but both halves"
        elif a["same_sign"] and min(a["p_H1"], a["p_H2"]) < 0.01:
            verdict = "Only one half"
        elif not a["same_sign"]:
            verdict = "Flips between halves"
        else:
            verdict = "Nothing"
        rows.append([html.escape(FEATURES[f]), f"{a['rho_H1']:+.2f}", f"{a['rho_H2']:+.2f}",
                     f"{t['rho_H1']:+.2f}", f"{t['rho_H2']:+.2f}", verdict])
    sections.append(("Everything else",
                     "Rank correlation (inside each week) with reaching +50% and with the take-profit-50 return. "
                     "+ means more of the feature, more winners. 'Holds' = same direction and p < 0.01 in both "
                     "halves.",
                     table(["Feature", "+50%: Jan-Jun", "+50%: Jul-Dec", "TP-50: Jan-Jun", "TP-50: Jul-Dec",
                            "Verdict"], rows, cls="corr")))

    # direction rules
    rows = [[html.escape(r["rule"]), r["stock-weeks"], p(r["chosen reached +50%"], 1), p(r["rejected reached +50%"], 1),
             f"{r['H1 edge TP-50 pts']:+.1f}", f"{r['H2 edge TP-50 pts']:+.1f}"] for r in s["direction_rules"]
            if not r["rule"].startswith(("Fade stock's 5", "Follow the opening", "Fade open"))]
    sections.append(("Picking the side (call or put)",
                     "Each stock-week had both a call and a put, so any rule for picking the side can be scored "
                     "against the side it rejects. Edge = take-profit-50 return of the chosen side minus the "
                     "rejected side, in percentage points.",
                     table(["Rule", "Stock-weeks", "Chosen reached +50%", "Rejected reached +50%",
                            "Edge Jan-Jun", "Edge Jul-Dec"], rows, cls="corr")))

    # models
    mo = s["models"]
    rows = []
    for k, v in mo.items():
        o, name, split = k.split("|")
        if name != "logistic":
            continue
        rows.append([{"win50": "Reached +50%", "dud": "Dud"}[o], split.replace("train ", "fit on ").replace(
            " -> test ", ", scored on "), f"{v['auc']:.2f}", p(v["test win50 top fifth"], 1),
            p(v["test win50 bottom fifth"], 1)])
    sections.append(("All features together",
                     "A model fitted on one half of the year and scored on the other. AUC 0.50 = coin flip, 1.00 = "
                     "perfect. Without the spread the same models score 0.50-0.52; the spread alone scores "
                     "0.58-0.66, better than everything combined.",
                     table(["Predicting", "Fit / scored", "AUC", "Top fifth reached +50%",
                            "Bottom fifth reached +50%"], rows, cls="corr")))
    notes = """<ul>
<li>Nothing knowable at 9:46 picks winners. The LSTM measures, volatility, beta, implied vol, earnings timing,
momentum, RSI, gaps and volume all failed to show up the same way in both halves of 2025.</li>
<li>The only reliable loser filter is cost: the wider the spread you pay, the more duds. That is arithmetic, not
prediction: bought at the ask and sold at the bid, a 30% spread has to be made back before any profit.</li>
<li>Following SPY's trend (the agent's own direction rule) and fading the opening gap worked strongly in
January-June, which held the April crash and rebound, and almost not at all in July-December.</li>
<li>Calls beat puts in both halves, but 2025 was a rising year; that would reverse in a falling one.</li>
<li>Even the best bucket (spread under 5%) only about broke even with a +50% take-profit.</li>
</ul>"""
    sections.append(("What it means", "", notes))
    body = "".join(
        f'<section><h2>{html.escape(h)}</h2>{f"<p class=sub>{html.escape(t)}</p>" if t else ""}{tb}</section>'
        for h, t, tb in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Winner And Loser Search</title>
<style>{CSS}</style></head><body><main>
<h1>Can we pick the winners, or weed out the losers?</h1>
<p class="lead">2025, 1,518 real option trades (Databento OPRA quotes). Generated from
results/analysis/calm_features.csv and winner_search.json.</p>
{body}
</main></body></html>"""
    out = R / "report" / "winner_search_tables.html"
    out.write_text(page, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
