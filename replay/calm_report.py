"""
Tables for the calm-stock test (results/analysis/calm_test.csv and
calm_summary.json), as one HTML page next to the replay tables.

    python -m replay.calm_report        # results/report/calm_test_tables.html
"""

from __future__ import annotations

import html
import json

import pandas as pd

from . import config
from .calm_analysis import load, quintiles
from .report_tables import CSS, table

R = config.RESULTS_DIR
# +1: a higher value means a more volatile stock; -1: a higher value means calmer.
ORIENT = {"avg_mae_pct": 1, "hit_rate_pct": -1, "vol_1y_pct": 1, "vol_20d_pct": 1, "beta_1y": 1, "iv_pct": 1}
NAMES = {"avg_mae_pct": "LSTM error (MAE)", "hit_rate_pct": "LSTM within ±2%", "vol_1y_pct": "1-year volatility",
         "vol_20d_pct": "20-day volatility", "beta_1y": "Stock beta", "iv_pct": "Implied vol at entry"}
OUTS = {"peak_pct": "Peak", "peak5_pct": "Peak within 5 days", "end_pct": "Held to Thursday exit"}


def p(x, d=0, sign=False):
    return "n/a" if pd.isna(x) else (f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%")


def qtable(d: pd.DataFrame, by: str) -> str:
    q = quintiles(d, by).reset_index()
    label = {"avg_mae_pct": "LSTM error", "vol_1y_pct": "1-year vol", "iv_pct": "implied vol"}[by]
    rows = []
    for _, r in q.iterrows():
        row = [html.escape(str(r.iloc[0])), int(r["options"]),
               (f"{r[f'median {by}']:.2f}%" if by == "avg_mae_pct" else p(r[f"median {by}"], 0))]
        if by != "iv_pct":
            row.append(p(r["median IV %"], 0))
        row += [f"{r['median leverage']:.0f}x", p(r["median biggest stock move %"], 1),
                f"<b>{p(r['median peak %'], 0, True)}</b>", p(r["peak >= +50%"], 0),
                p(r["median at exit %"], 0, True)]
        rows.append(row)
    head = ["Fifth", "Options", f"Median {label}"] + (["Implied vol"] if by != "iv_pct" else []) + \
           ["Leverage", "Biggest stock move", "Median peak", "Reached +50%", "At Thursday exit (median)"]
    return table(head, rows)


def weekly_table(s: dict, keys: list[tuple[str, str]]) -> str:
    rows = []
    for x, y in keys:
        t = s["tests"].get(f"{x}|{y}")
        if not t:
            continue
        calm = t["weeks_negative"] if ORIENT[x] == 1 else t["weeks_positive"]
        vol = t["weeks_positive"] if ORIENT[x] == 1 else t["weeks_negative"]
        rows.append([NAMES[x], OUTS[y], calm, vol, f"{t['median_weekly_rho'] * ORIENT[x] * -1:+.2f}",
                     f"{t['wilcoxon_p']:.3f}" if t["wilcoxon_p"] is not None else "n/a"])
    return table(["Calmness measure", "Outcome", "Weeks calmer stocks did better", "Weeks volatile stocks did better",
                  "Median weekly correlation (+ = calmer better)", "p-value"], rows, cls="corr")


def main():
    d = load()
    raw = pd.read_csv(R / "analysis" / "calm_test.csv")
    s = json.loads((R / "analysis" / "calm_summary.json").read_text())
    sections = []

    skips = raw["skip"].value_counts()
    rows = [["Options measured", f"{len(d):,}"],
            ["Stocks", f"{d['ticker'].nunique()}"],
            ["Weeks", f"{d['asof'].nunique()}"],
            ["Stage-1 candidates / random S&P 500 controls", f"{(d['group'] == 'candidate').sum()} / "
                                                             f"{(d['group'] == 'control').sum()} options"],
            ["Agent-tradable (spread ≤ 20%, delta ≥ 0.35)", f"{d['tradable'].sum()} options"],
            ["Median peak (all options)", p(d["peak_pct"].median(), 0, True)],
            ["Reached +50% at some point", p(d["hit50"].mean() * 100, 0)],
            ["Median / mean value at the Thursday exit", f"{p(d['end_pct'].median(), 0, True)} / "
                                                         f"{p(d['end_pct'].mean(), 0, True)}"],
            ["Skipped (the agent couldn't have bought them)", "; ".join(f"{k}: {v}" for k, v in skips.items())]]
    sections.append(("The sample",
                     "Each week: every stage-1 candidate plus 10 random S&P 500 members. On the first trading day "
                     "of the following week at 9:46 ET, the call and the put the agent's rule would pick "
                     "(12-24 days to expiry, strike about 1% out of the money) were bought at the ask, then "
                     "followed on real bid quotes until the agent's Thursday-before-expiry exit.",
                     table(["", "Value"], rows)))
    sections.append(("By LSTM error", "Stocks ranked by stage-1 LSTM error inside their own week, then split "
                     "into fifths. Peak = best bid reached / price paid.", qtable(d, "avg_mae_pct")))
    sections.append(("By 1-year volatility", "", qtable(d, "vol_1y_pct")))
    sections.append(("By implied volatility at entry", "", qtable(d, "iv_pct")))

    keys = [(x, y) for x in ["avg_mae_pct", "hit_rate_pct", "vol_1y_pct", "iv_pct"] for y in OUTS]
    sections.append(("Week by week, all options",
                     "Inside each week, the correlation between the measure and the outcome. Counting weeks "
                     "avoids treating ~1,500 overlapping options as independent. p-value: Wilcoxon test that "
                     "the weekly correlations centre on zero (below 0.05 is unlikely to be chance).",
                     weekly_table(s["all"], keys)))
    rows = []
    for k in ["all", "tradable", "candidates", "controls", "calls", "puts"]:
        t = s[k]["tests"].get("avg_mae_pct|peak_pct")
        if t:
            rows.append([html.escape(s[k]["label"].replace(" (spread <= 20%, delta >= 0.35)", "")), s[k]["options"], t["weeks_negative"], t["weeks_positive"],
                         f"{-t['median_weekly_rho']:+.2f}", f"{t['wilcoxon_p']:.3f}"])
    sections.append(("LSTM error vs peak, by subset", "",
                     table(["Subset", "Options", "Weeks calmer stocks did better", "Weeks volatile stocks did better",
                            "Median weekly correlation (+ = calmer better)", "p-value"], rows, cls="corr")))
    pr = s["partial"]
    notes = f"""<ul>
<li>Calmer, lower-error stocks did not give bigger option peaks. Across all options the higher-error fifth peaked
slightly higher, not lower; holding to the Thursday exit showed no difference at all.</li>
<li>The reason: calm stocks have cheaper options with more leverage, but they also move less, and the two cancel out.
Sorted by implied vol, the median peak is flat.</li>
<li>Once implied vol is known, LSTM error adds nothing (partial correlation {pr['partial_rho_mae_peak_given_iv']:+.2f}).</li>
<li>Among the stage-1 candidates alone, the names the funnel actually trades, there is no relationship either way.</li>
<li>Calls on more volatile stocks did better in 2025, a rising year; puts showed no such pattern. That is the market's
direction, not calmness.</li>
<li>The earlier pattern (calmer names gave the big peaks) came from 7 trades on 5 stocks and does not hold up.</li>
</ul>"""
    sections.append(("What it means", "", notes))

    body = "".join(
        f'<section><h2>{html.escape(h)}</h2>{f"<p class=sub>{html.escape(t)}</p>" if t else ""}{tb}</section>'
        for h, t, tb in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calm-Stock Test</title>
<style>{CSS}</style></head><body><main>
<h1>Do calmer, lower-error stocks give bigger option peaks?</h1>
<p class="lead">2025, real OPRA option quotes (Databento). Generated from results/analysis/calm_test.csv.</p>
{body}
</main></body></html>"""
    out = R / "report" / "calm_test_tables.html"
    out.write_text(page, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
