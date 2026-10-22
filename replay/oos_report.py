"""
Results page for the pre-registered out-of-sample spread test
(results/oos/verdict.json, oos_trades.csv), with the 2025 discovery numbers
alongside for comparison.

    python -m replay.oos_report      # results/report/oos_spread_test.html
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from . import config
from .oos_spread import MIN_DELTA, PERIODS, SPREAD_MAX_PCT, TRADES, VERDICT, week_boot
from .report_tables import CSS, table

R = config.RESULTS_DIR
NAMES = {"2024": "2024 (52 weeks)", "2026": "Jan-Sep 2026 (39 weeks)"}
CRIT = {"1_mean_and_ci_above_zero": "1. Average above zero, 90% range above zero",
        "2_both_halves_positive": "2. Positive in both halves",
        "3_positive_without_top_3pct": "3. Positive without the best 3% of trades",
        "4_positive_without_best_month": "4. Positive without the best month"}


def p(x, d=1, sign=True):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%")


def ok(b):
    return "<b>pass</b>" if b else "fail"


def discovery_2025() -> dict:
    d = pd.read_csv(R / "analysis" / "spread_check.csv")
    s = d[(d["spread_pct"] < SPREAD_MAX_PCT) & (d["delta"] >= MIN_DELTA)]
    lo, hi = week_boot(s, "v3")
    return {"n": len(s), "weeks": s["asof"].nunique(), "mean": s["v3"].mean(), "ci90": [lo, hi],
            "h1": s.loc[s["half"] == "H1", "v3"].mean(), "h2": s.loc[s["half"] == "H2", "v3"].mean(),
            "wide": d[d["tradable"] & (d["spread_pct"] >= SPREAD_MAX_PCT)]["v3"].mean()}


def main():
    v = json.loads(VERDICT.read_text())
    t = pd.read_csv(TRADES)
    t["period"] = t["period"].astype(str)
    q = t[(t["qualifies"] == True) & t["v3_pct"].notna()].copy()  # noqa: E712
    d25 = discovery_2025()
    sections = []

    sections.append(("The question", (
        "In 2025, options bought with a bid-ask spread under 5% at entry made money under the live V3 sell "
        "rules, but that 5% line was drawn on the same 2025 data. This page is the test of the unchanged rule on "
        "two periods it had never seen, scored against criteria written down before the data was downloaded "
        "(results/oos/PREREGISTRATION.md)."),
        table(["", "Rule"], [
            ["Entry", "First trading day of each week, 9:46 ET, at the ask"],
            ["Contract", "The agent's pick: 12-24 days to expiry (14-21 preferred), strike nearest 1% out of the money; "
                         "both the call and the put"],
            ["Qualifies", f"Spread under {SPREAD_MAX_PCT:.0f}% of the mid and delta at least {MIN_DELTA}"],
            ["Exit", "Live V3 rules every minute: stop -75%, ceiling +200%, trail 20 points after +50%, "
                     "force exit 13:00 the day before expiry; trigger on the mid, sell at the bid"],
            ["Stocks", "Each week the 40 S&P 500 members with the most dollar volume over the prior 20 days, "
                       "plus 10 random members"]])))

    rows = [["2025 (where the lead was found)", d25["n"], d25["weeks"], f"<b>{p(d25['mean'])}</b>",
             f"{p(d25['ci90'][0])} to {p(d25['ci90'][1])}", p(d25["h1"]), p(d25["h2"]), "", ""]]
    for per, x in v["periods"].items():
        if x.get("verdict") == "underpowered":
            rows.append([NAMES[per], x["n"], x["weeks"], "underpowered", "", "", "", "", "underpowered"])
            continue
        rows.append([NAMES[per], x["n"], x["weeks"], f"<b>{p(x['mean'])}</b>",
                     f"{p(x['ci90'][0])} to {p(x['ci90'][1])}", p(x["h1"]["mean"]), p(x["h2"]["mean"]),
                     p(x["median"]), f"<b>{x['verdict']}</b>"])
    sections.append((f"Verdict: {v['verdict']}",
                     "Average return per qualifying trade under the live V3 sell rules (bought at the ask, sold at "
                     "the bid). The 90% range resamples whole weeks.",
                     table(["Period", "Trades", "Weeks", "Average", "90% range", "1st half", "2nd half",
                            "Median", "Verdict"], rows)))

    rows = []
    for key, label in CRIT.items():
        r = [label]
        for per, x in v["periods"].items():
            c = x.get("criteria")
            if not c:
                r.append("n/a"); continue
            extra = {"1_mean_and_ci_above_zero": f"{p(x['mean'])}, low end {p(x['ci90'][0])}",
                     "2_both_halves_positive": f"{p(x['h1']['mean'])} / {p(x['h2']['mean'])}",
                     "3_positive_without_top_3pct": f"{p(x['drop_top']['mean'])} without {x['drop_top']['k']} trades",
                     "4_positive_without_best_month": f"{p(x['best_month']['mean_without'])} without "
                                                      f"{x['best_month']['month']}"}[key]
            r.append(f"{ok(c[key])}: {extra}")
        rows.append(r)
    sections.append(("The four pre-registered checks", "A period passes only if all four pass.",
                     table(["Check", *[NAMES[k] for k in v["periods"]]], rows)))

    rows = []
    for per, x in v["periods"].items():
        s = x["secondary"]
        r = s["rules_on_qualifying"]
        rows.append([NAMES[per], p(r["v3"]["mean"]), p(r["v3_plus_day2"]["mean"]), p(r["day2_only"]["mean"]),
                     p(r["hold_to_thursday"]["mean"]), p(r["take_profit_50"]["mean"]),
                     f"{p(s['wide_5_to_20_v3']['mean'])} ({s['wide_5_to_20_v3']['n']})"])
    rows.append(["2025 (discovery)", p(d25["mean"]), "", "", "", "", p(d25["wide"])])
    sections.append(("Other sell rules on the same trades, and the wider spreads",
                     "Reported, not part of the verdict. The last column is the 5-20% spread options from the same "
                     "sample under V3 (trades in brackets).",
                     table(["Period", "V3", "V3 + day-2 sell", "Day-2 sell only", "Hold to Thursday",
                            "Take profit +50%", "5-20% spreads, V3"], rows)))

    rows = []
    for per, x in v["periods"].items():
        s = x["secondary"]
        rows.append([NAMES[per],
                     f"{p(s['by_group']['liquid']['mean'])} ({s['by_group']['liquid']['n']})",
                     f"{p(s['by_group']['random']['mean'])} ({s['by_group']['random']['n']})",
                     f"{p(s['by_side']['CALL']['mean'])} ({s['by_side']['CALL']['n']})",
                     f"{p(s['by_side']['PUT']['mean'])} ({s['by_side']['PUT']['n']})",
                     f"{p(s['one_side_follow_spy_trend']['mean'])} ({s['one_side_follow_spy_trend']['n']})"])
    sections.append(("Splits", "Average V3 return, trades in brackets. 'Follow SPY' keeps only the call when SPY is "
                     "above its 20-day average and only the put otherwise, one trade per stock-week.",
                     table(["Period", "Liquid 40", "Random 10", "Calls", "Puts", "Follow SPY trend"], rows)))

    cuts = ["spread<3", "spread<4", "spread<5", "spread<6", "spread<8", "spread<10"]
    rows = [[NAMES[per]] + [f"{p(x['secondary']['cutoff_sensitivity'][c]['mean'])} "
                            f"({x['secondary']['cutoff_sensitivity'][c]['n']})" for c in cuts]
            for per, x in v["periods"].items()]
    sections.append(("Where the spread line is drawn", "Average V3 return for all measured options under each "
                     "spread cut-off (delta at least 0.35). Only the 5% line is the test; the others show whether "
                     "the result depends on exactly where the line sits.",
                     table(["Period", *[c.replace("spread<", "under ") + "%" for c in cuts]], rows)))

    q["month"] = q["entry_date"].str[:7]
    rows = []
    for per in PERIODS:
        g = q[q["period"] == per].groupby("month")["v3_pct"]
        for mth, s in g:
            rows.append([mth, len(s), p(s.mean()), p(s.median()), f"{(s > 0).mean() * 100:.0f}%"])
    sections.append(("Month by month", "Qualifying trades by entry month, V3 return.",
                     table(["Month", "Trades", "Average", "Median", "Winners"], rows, cls="corr")))

    rows = []
    for per, x in v["periods"].items():
        for r in x["secondary"]["top_tickers"][:12]:
            rows.append([NAMES[per], html.escape(r["ticker"]), r["count"], p(r["mean"])])
    sections.append(("Most-traded names", "", table(["Period", "Ticker", "Trades", "Average V3"], rows, cls="corr")))

    body = "".join(
        f'<section><h2>{html.escape(h)}</h2>{f"<p class=sub>{html.escape(tx)}</p>" if tx else ""}{tb}</section>'
        for h, tx, tb in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Out-Of-Sample Spread Test</title>
<style>{CSS}</style></head><body><main>
<h1>Does the tight-spread rule hold up on new data?</h1>
<p class="lead">Pre-registered test on 2024 and January-September 2026, real OPRA quotes (Databento), every
exit simulated minute by minute. Generated from results/oos/verdict.json and oos_trades.csv.</p>
{body}
</main></body></html>"""
    out = R / "report" / "oos_spread_test.html"
    out.write_text(page, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
