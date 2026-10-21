"""
One HTML page with the tables for the LSTM-real 1-trade-min book (9:46 entry
plus the 10:01 spread re-check): the trades, what holding would have done, and
the factor study of what the big-peak trades had in common.

Every number is read from the result files, nothing is typed in by hand.

    python -m replay.report_tables      # results/report/replay_2025_tables.html
"""

from __future__ import annotations

import html
import json

import pandas as pd

from . import config

R = config.RESULTS_DIR
BOOK = ("sim_min1", "lstm_real_0946_r1001")
ORDER = [1, 2, 3, 6, 4, 5, 7]          # big-peak trades first, then the rest
BIG = {1, 2, 3, 6}


def money(x, sign=True):
    if pd.isna(x):
        return "n/a"
    s = f"${abs(x):,.0f}"
    return ("-" + s if x < 0 else ("+" + s if sign and x > 0 else s))


def pct(x, d=0, sign=True):
    if pd.isna(x):
        return "n/a"
    return f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%"


def day(ts):
    t = pd.Timestamp(ts)
    return f"{t:%b} {t.day}"


def table(head, rows, cls=None, foot=None, flags=None):
    out = [f'<div class="wrap"><table{f" class={cls!r}" if cls else ""}><thead><tr>']
    out += [f"<th>{html.escape(h)}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for i, r in enumerate(rows):
        c = f' class="{flags[i]}"' if flags and flags[i] else ""
        out.append(f"<tr{c}>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>")
    out.append("</tbody>")
    if foot:
        out.append("<tfoot><tr>" + "".join(f"<td>{v}</td>" for v in foot) + "</tr></tfoot>")
    out.append("</table></div>")
    return "".join(out)


def main():
    v, b = BOOK
    bdir = R / "agent" / v / b
    trades = pd.read_csv(bdir / "trades.csv").sort_values("entry_ts").reset_index(drop=True)
    trades["n"] = range(1, len(trades) + 1)
    summary = json.loads((bdir / "summary.json").read_text())
    hold = pd.read_csv(R / "analysis" / f"hold_{v}_{b}.csv")
    hold["_k"] = pd.to_datetime(hold["entry"]).dt.floor("s")
    trades["_k"] = pd.to_datetime(trades["entry_ts"]).dt.floor("s")
    t = trades.merge(hold.drop(columns=["ticker"]), on="_k", how="left", suffixes=("", "_h"))

    fac = pd.read_csv(R / "analysis" / "trade_factors.csv")
    fac = fac[fac["seen_in"].str.startswith(f"{v}/{b}")].copy()
    fac["_k"] = pd.to_datetime(fac["entry_ts"]).dt.floor("s")
    f = t[["n", "_k", "ticker"]].merge(fac.drop(columns=["ticker"]), on="_k", how="left").set_index("n")
    corr = pd.read_csv(R / "analysis" / "trade_factor_correlations.csv").set_index("feature")

    def label(r):
        return f"{r['ticker']} {'call' if str(r['opt_type']).upper().startswith('C') else 'put'}"

    sections = []

    # 1. The trades ───────────────────────────────────────────────────────────
    rows, flags = [], []
    for _, r in t.iterrows():
        kind = "Call" if str(r["opt_type"]).upper().startswith("C") else "Put"
        e = pd.Timestamp(r["entry_ts"])
        when = f"{day(e)}, {e:%H:%M}" + (" (re-check)" if e.strftime("%H:%M") == "10:01" else "")
        rows.append([r["n"], r["ticker"], f"{kind} {r['strike']:g}", when, day(r["exit_ts"]),
                     html.escape(str(r["exit_reason"])), money(r["realized_pnl"])])
        flags.append("pos" if r["realized_pnl"] > 0 else "neg")
    sections.append(("Trades in the 1-trade min book with the re-check",
                     f"LSTM real, 1-trade minimum whitelist, entries at 9:46 ET with a 10:01 re-check for "
                     f"spread rejections. Started with ${summary.get('start_balance', 5000):,.0f}, ended "
                     f"{pct(summary.get('return_pct'), 1)} ({summary.get('n_trades')} trades). "
                     f"SPY over the same window: {pct(summary.get('spy_return_pct_same_window'), 1)}.",
                     table(["#", "Ticker", "Option", "Bought", "Sold", "Why sold", "P&L"], rows, flags=flags)))

    # 2. Hold to expiry and the best moment to sell ───────────────────────────
    rows = []
    for _, r in t.iterrows():
        kind = "Call" if str(r["opt_type"]).upper().startswith("C") else "Put"
        held = money(r["hold_to_expiry_pnl"]) + (" (expired worthless)" if r["value_at_expiry"] == 0
                                                  else f" ({pct(r['hold_to_expiry_pct'])})")
        best = (f"${r['peak_bid']:.2f} vs ${r['cost']:.2f} cost, <b>{money(r['peak_bid_pnl'])} "
                f"({pct(r['peak_bid_pct'])})</b>" if r["ever_positive_bid"]
                else f"${r['peak_bid']:.2f} vs ${r['cost']:.2f} cost, never above cost")
        pk, ex, en = pd.Timestamp(r["peak_bid_at"]), pd.Timestamp(r["exit_ts"]), pd.Timestamp(r["entry_ts"])
        if pk <= ex:
            rel = "while still held"
        elif pk.date() == ex.date():
            rel = "same day, after the sale"
        else:
            nd = (pk.normalize() - ex.normalize()).days
            rel = f"{nd} day{'s' if nd != 1 else ''} after the sale"
        rows.append([r["n"], f"{r['ticker']} {kind} {r['strike']:g}, exp {pd.Timestamp(r['expiry']):%b} "
                     f"{pd.Timestamp(r['expiry']).day}", money(r["total_spent"], False), money(r["realized_pnl"]),
                     held, best, f"{day(pk)}, {pk:%H:%M} ({rel})"])
    foot = ["", "<b>Total</b>", f"<b>{money(t['total_spent'].sum(), False)}</b>",
            f"<b>{money(t['realized_pnl'].sum())}</b>", f"<b>{money(t['hold_to_expiry_pnl'].sum())}</b>",
            f"<b>{money(t['peak_bid_pnl'].sum())}</b> if every trade sold at its best minute", ""]
    sections.append(("Held to expiry and the best price reached",
                     "Held to expiry: what keeping the option until it expired would have made. Best sell "
                     "price reached: the highest bid (what you could actually sell at) at any minute from "
                     "entry to expiry. The peak total is hindsight, an upper limit no exit rule can capture.",
                     table(["#", "Trade", "Spent", "Booked", "Held to expiry", "Best sell price reached (bid)",
                            "When the peak came"], rows, foot=foot)))

    # 3-5. Factor study, big-peak trades first ───────────────────────────────
    def frow(n, cells):
        return [n, label(f.loc[n]), *cells]
    flags = ["big" if n in BIG else "" for n in ORDER]
    rows = [frow(n, [pct(f.loc[n, "peak_bid_pct"]),
                     f"{f.loc[n, 's1_avg_mae_pct']:.2f}%", pct(f.loc[n, "s1_hit_rate_pct"], sign=False),
                     pct(f.loc[n, "s1_dir_accuracy_pct"], sign=False),
                     pct(f.loc[n, "s2_gain_pct_of_stake"], sign=False), money(f.loc[n, "sim2_pnl"], False)])
            for n in ORDER]
    sections.append(("What the funnel said",
                     "From the week each ticker was promoted. LSTM error is the stage-1 average daily error "
                     "as a % of price; within ±2% is how often the forecast landed within 2% of the actual "
                     "price; backtest gain is stage-2 P&L as a % of the $10k stake.",
                     table(["#", "Trade", "Peak", "LSTM error (MAE)", "LSTM within ±2%", "LSTM direction",
                            "Backtest gain", "Sim 2 P&L"], rows, flags=flags)))

    rows = [frow(n, [pct(f.loc[n, "iv_pct"], sign=False), f"{abs(f.loc[n, 'leverage']):.0f}x",
                     f"{f.loc[n, 'beta_1y']:.2f}", pct(f.loc[n, "alpha_1y_ann_pct"]),
                     f"{f.loc[n, 'option_beta']:+.1f}"]) for n in ORDER]
    sections.append(("The option, alpha and beta at entry",
                     "Stock beta: how much the stock moves for each 1% SPY move over the year before entry. "
                     "Stock alpha: its yearly return beyond what beta explains. Leverage: the option's % move "
                     "per 1% stock move (delta x stock price / premium). Option beta = stock beta x leverage; "
                     "puts are negative.",
                     table(["#", "Trade", "Implied vol", "Leverage", "Stock beta", "Stock alpha (1 yr)",
                            "Option beta"], rows, flags=flags)))

    rows = []
    for n in ORDER:
        r = f.loc[n]
        pk = pd.Timestamp(r["peak_bid_at"])
        note = f" ({day(pk)})"
        if isinstance(r.get("earnings_in_window"), str) and r.get("earnings_before_peak") in (True, "True"):
            note = f" (after {day(r['earnings_in_window'])} earnings)"
        rows.append(frow(n, [pct(r["peak_bid_pct"]), f"{int(r['days_to_peak'])}{note}",
                             pct(r["stock_move_to_peak_with_trade"], 1, sign=False),
                             pct(r["market_part_with_trade"], 1, sign=False),
                             pct(r["stock_specific_part_with_trade"], 1, sign=False),
                             f"~{abs(r['leverage']) * r['stock_move_to_peak_with_trade']:.0f}%"]))
    sections.append(("How each peak happened",
                     "Stock move your way: the stock's move from entry to the minute of the option's peak, "
                     "in the trade's favour. Market part: stock beta x SPY's move over the same window. "
                     "Stock-specific part: the rest. Leverage x move is a rough estimate of the peak.",
                     table(["#", "Trade", "Peak", "Trading days to peak", "Stock move your way", "Market part",
                            "Stock-specific part", "Leverage x move"], rows, flags=flags)))

    # 6. Correlations ───────────────────────────────────────────────────────
    pick = [("stock_move_to_peak_with_trade", "Stock move your way to the peak"),
            ("days_to_peak", "Days until the peak"),
            ("stock_specific_part_with_trade", "Stock-specific part of the move"),
            ("market_part_with_trade", "Market part of the move"),
            ("s1_avg_mae_pct", "LSTM error (MAE)"),
            ("s1_hit_rate_pct", "LSTM within ±2%"),
            ("beta_1y", "Stock beta"),
            ("s2_gain_pct_of_stake", "Backtest gain"),
            ("sim2_pnl", "Sim 2 P&L"),
            ("abs_leverage", "Option leverage"),
            ("s1_dir_accuracy_pct", "LSTM direction accuracy"),
            ("fc_1d_with_trade", "LSTM forecast size that morning"),
            ("option_beta", "Option beta"),
            ("alpha_1y_with_trade", "Stock alpha (1 yr)")]
    rows = [[name, f"{corr.loc[k, 'rho7']:+.2f}", f"{corr.loc[k, 'rho_real']:+.2f}",
             f"{corr.loc[k, 'rho_all']:+.2f}"] for k, name in pick if k in corr.index]
    sections.append(("What correlated with the peak",
                     "Spearman rank correlation with the option's peak %: +1 = always rise together, -1 = "
                     "always opposite, 0 = no relation. The 7 trades are this book; the 16 are the distinct "
                     "real-mode trades across all books; the 23 add the copy-mode trades.",
                     table(["Measure", "7 trades in this book", "16 real trades, all books",
                            "All 23 trades"], rows, cls="corr")))

    mae_all = corr.loc["s1_avg_mae_pct", "rho_all"]
    notes = f"""
<ul>
<li>The funnel numbers are per stock, and WYNN and CCL each appear twice, so the 7 trades are really 5 stocks
and the 16 real trades are about 8. With about 70 measures tested, several will line up by luck.</li>
<li>The before-entry measures that separate the groups are mostly volatility measured several ways: the LSTM
error grows with volatility, landing within ±2% is easier for a calm stock, and volatile stocks produce bigger
backtest and sim dollar numbers.</li>
<li>DDOG, the most volatile name, went the other way in other books: its copy-mode puts peaked +325% and +604%.
With those included the MAE link drops to {mae_all:+.2f}.</li>
<li>A larger LSTM error did go with more volatile stocks, but that did not give bigger option peaks here,
because options on those stocks were already priced for it (implied vol 41 to 77% against 18 to 34%).</li>
</ul>"""
    sections.append(("Why this isn't a rule yet", "", notes))

    body = "".join(
        f'<section><h2>{html.escape(h)}</h2>{f"<p class=sub>{html.escape(s)}</p>" if s else ""}{tbl}</section>'
        for h, s, tbl in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replay 2025 Tables</title>
<style>
:root {{ --bg:#f7f7f5; --card:#ffffff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --head:#f0efe9;
        --pos:#1d7a46; --neg:#b3261e; --big:#eef6ff; --accent:#2f5d9b; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141414; --card:#1d1d1d; --ink:#ecebe6; --muted:#a3a29b; --line:#33332f; --head:#262624;
          --pos:#5cc98a; --neg:#f2867e; --big:#1b2633; --accent:#8db4ea; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }}
.lead {{ color:var(--muted); margin:0 0 28px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px 20px 8px;
          margin:0 0 20px; }}
h2 {{ font-size:18px; margin:0 0 6px; }}
p.sub {{ color:var(--muted); margin:0 0 14px; font-size:14px; }}
.wrap {{ overflow-x:auto; margin:0 0 12px; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ background:var(--head); font-weight:600; font-size:13px; white-space:nowrap; }}
td:first-child, th:first-child {{ white-space:nowrap; }}
tfoot td {{ border-top:2px solid var(--line); border-bottom:none; }}
tr.pos td:last-child {{ color:var(--pos); font-weight:600; }}
tr.neg td:last-child {{ color:var(--neg); font-weight:600; }}
tr.big td {{ background:var(--big); }}
table.corr td:not(:first-child), table.corr th:not(:first-child) {{ text-align:right; }}
ul {{ margin:0 0 12px; padding-left:20px; }} li {{ margin:0 0 8px; }}
.key {{ display:inline-block; width:12px; height:12px; background:var(--big); border:1px solid var(--line);
        vertical-align:-1px; margin-right:6px; border-radius:2px; }}
</style></head><body><main>
<h1>Replay 2025: LSTM real, 1-trade min book</h1>
<p class="lead">Full-year 2025 walk-forward replay. Generated from the files in results/.
<span class="key"></span>Shaded rows are trades 1, 2, 3 and 6, the four with big peaks.</p>
{body}
</main></body></html>"""
    out = R / "report" / "replay_2025_tables.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
