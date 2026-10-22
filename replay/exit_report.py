"""
Sell-strategy report: a chart for every agent trade, the common patterns across
the 1,518 calm-test options, and how each sell rule would have done.

    python -m replay.exit_report      # results/report/sell_strategy.html
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from . import asof, config
from .report_tables import CSS, table

R = config.RESULTS_DIR
DAYS = asof.trading_days(pd.Timestamp("2025-01-01").date(), pd.Timestamp("2026-02-27").date())
DIX = {d: i for i, d in enumerate(DAYS)}

VIZ_CSS = """
.viz-root { --s1:#2a78d6; --s2:#eb6834; --band1:rgba(42,120,214,.12); --band2:rgba(235,104,52,.12);
  --grid:#e1e0d9; --axis:#c3c2b7; --muted:#898781; --ink:#0b0b0b; --ink2:#52514e; --surf:#fcfcfb; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
  --s1:#3987e5; --s2:#d95926; --band1:rgba(57,135,229,.16); --band2:rgba(217,89,38,.16);
  --grid:#2c2c2a; --axis:#383835; --muted:#898781; --ink:#f0efec; --ink2:#c3c2b7; --surf:#1a1a19; } }
:root[data-theme="dark"] .viz-root { --s1:#3987e5; --s2:#d95926; --band1:rgba(57,135,229,.16);
  --band2:rgba(217,89,38,.16); --grid:#2c2c2a; --axis:#383835; --ink:#f0efec; --ink2:#c3c2b7; --surf:#1a1a19; }
.viz-root svg { display:block; width:100%; height:auto; overflow:visible; }
.viz-root text { font: 11px -apple-system, "Segoe UI", Roboto, sans-serif; fill: var(--muted); }
.viz-root text.lbl { fill: var(--ink2); font-weight:600; }
.multiples { display:grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap:14px; }
.mini { border:1px solid var(--line); border-radius:10px; padding:10px 10px 4px; background:var(--card); }
.mini h3 { font-size:13px; margin:0 0 2px; } .mini p { font-size:12px; color:var(--muted); margin:0 0 6px; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--muted); margin:0 0 8px; flex-wrap:wrap; }
.legend i { display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; }
.legend b { display:inline-block; width:9px; height:9px; border-radius:50%; vertical-align:-1px; margin-right:6px; }
.tip { position:fixed; pointer-events:none; background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:6px 9px; font-size:12px; box-shadow:0 4px 14px rgba(0,0,0,.12); display:none; z-index:9; color:var(--ink); }
.tip .v { font-weight:700; } .tip .k { color:var(--muted); }
"""

HOVER_JS = """
<div class="tip" id="tip"></div>
<script>
(function(){
  const tip = document.getElementById('tip');
  document.querySelectorAll('svg[data-hover]').forEach(svg => {
    let d; try { d = JSON.parse(svg.getAttribute('data-hover')); } catch(e) { return; }
    const ns = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('stroke', 'var(--axis)'); line.setAttribute('stroke-width', '1');
    line.setAttribute('y1', d.top); line.setAttribute('y2', d.bottom); line.style.display = 'none';
    svg.appendChild(line);
    svg.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(); const vb = svg.viewBox.baseVal;
      const px = (ev.clientX - r.left) * vb.width / r.width;
      if (px < d.left || px > d.right) { tip.style.display='none'; line.style.display='none'; return; }
      const xv = d.x0 + (px - d.left) / (d.right - d.left) * (d.x1 - d.x0);
      tip.textContent = '';
      const head = document.createElement('div'); head.className = 'k';
      head.textContent = d.xfmt.replace('{x}', xv.toFixed(1)); tip.appendChild(head);
      let snap = null;
      d.series.forEach(s => {
        let best = null, bd = 1e9;
        s.pts.forEach(p => { const dd = Math.abs(p[0] - xv); if (dd < bd) { bd = dd; best = p; } });
        if (!best) return; snap = best[0];
        const row = document.createElement('div');
        const v = document.createElement('span'); v.className = 'v';
        v.textContent = (best[1] > 0 ? '+' : '') + best[1].toFixed(0) + '%';
        const k = document.createElement('span'); k.className = 'k'; k.textContent = '  ' + s.name;
        row.appendChild(v); row.appendChild(k); tip.appendChild(row);
      });
      if (snap === null) return;
      const sx = d.left + (snap - d.x0) / (d.x1 - d.x0) * (d.right - d.left);
      line.setAttribute('x1', sx); line.setAttribute('x2', sx); line.style.display = '';
      tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
    });
    svg.addEventListener('pointerleave', () => { tip.style.display='none'; line.style.display='none'; });
  });
})();
</script>"""


def nice_ticks(lo, hi, n=5):
    span = max(hi - lo, 1e-9)
    step = 10 ** np.floor(np.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    return np.arange(np.floor(lo / step) * step, hi + step * 0.5, step)


def line_chart(series, w=640, h=260, x_label="trading days after buying", ymin=None, ymax=None,
               markers=(), xmax=None, compact=False):
    """series: [{name, color, pts: [(x,y)], band: [(x,lo,hi)] or None}]"""
    L, Rr, T, B = (34, 10, 8, 22) if compact else (44, 14, 10, 30)
    xs = [p[0] for s in series for p in s["pts"]]
    ys = [p[1] for s in series for p in s["pts"]] + [b for s in series for p in (s.get("band") or []) for b in p[1:]]
    ys += [m["y"] for m in markers]
    x0, x1 = 0, xmax if xmax is not None else max(xs + [1])
    lo = min(ys + [0]) if ymin is None else ymin
    hi = max(ys + [0]) if ymax is None else ymax
    yt = nice_ticks(lo, hi, 4 if compact else 5)
    lo, hi = min(lo, yt[0]), max(hi, yt[-1])
    X = lambda x: L + (x - x0) / (x1 - x0) * (w - L - Rr)
    Y = lambda y: T + (hi - y) / (hi - lo) * (h - T - B)
    out = []
    for t in yt:
        out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:+.0f}%</text>'
                   if t else f'<text x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">0</text>')
    out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    for t in range(0, int(np.ceil(x1)) + 1, 1 if x1 <= 12 else 2):
        if t <= x1:
            out.append(f'<text x="{X(t):.1f}" y="{h - B + 14}" text-anchor="middle">{t}</text>')
    if not compact:
        out.append(f'<text x="{(L + w - Rr) / 2:.0f}" y="{h - 2}" text-anchor="middle">{html.escape(x_label)}</text>')
    for s in series:
        if s.get("band"):
            top = " ".join(f"{X(x):.1f},{Y(hi_):.1f}" for x, lo_, hi_ in s["band"])
            bot = " ".join(f"{X(x):.1f},{Y(lo_):.1f}" for x, lo_, hi_ in reversed(s["band"]))
            out.append(f'<polygon points="{top} {bot}" fill="var(--band{s["color"]})" stroke="none"/>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["pts"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{s["color"]})" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
    for m in markers:
        out.append(f'<circle cx="{X(m["x"]):.1f}" cy="{Y(m["y"]):.1f}" r="4.5" fill="var(--s{m["color"]})" '
                   f'stroke="var(--surf)" stroke-width="2"/>')
        anchor = "end" if X(m["x"]) > w * 0.75 else "start"
        dx = -7 if anchor == "end" else 7
        dy = m.get("dy", -7)
        out.append(f'<text class="lbl" x="{X(m["x"]) + dx:.1f}" y="{Y(m["y"]) + dy:.1f}" '
                   f'text-anchor="{anchor}">{html.escape(m["label"])}</text>')
    hover = {"left": L, "right": w - Rr, "top": T, "bottom": h - B, "x0": x0, "x1": x1,
             "xfmt": "day {x} after buying",
             "series": [{"name": s["name"], "pts": [[round(x, 3), round(y, 1)] for x, y in s["pts"]]} for s in series]}
    return (f'<svg viewBox="0 0 {w} {h}" role="img" data-hover=\'{html.escape(json.dumps(hover))}\'>'
            + "".join(out) + "</svg>")


def bar_chart(cats, vals, w=640, h=220, fmt="{:.0f}%", color=1, x_label=""):
    L, Rr, T, B = 40, 10, 16, 34
    hi = max(vals) * 1.15 if len(vals) else 1
    yt = nice_ticks(0, hi, 4)
    hi = yt[-1]
    n = len(cats)
    slot = (w - L - Rr) / n
    bw = min(24, slot * 0.6)
    Y = lambda y: T + (hi - y) / hi * (h - T - B)
    out = []
    for t in yt:
        out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{fmt.format(t)}</text>')
    for i, (c, v) in enumerate(zip(cats, vals)):
        x = L + slot * i + (slot - bw) / 2
        y = Y(v)
        hgt = Y(0) - y
        r = min(4, hgt / 2)
        out.append(f'<path d="M{x:.1f},{Y(0):.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
                   f'H{x + bw - r:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{Y(0):.1f} Z" '
                   f'fill="var(--s{color})"><title>{html.escape(str(c))}: {fmt.format(v)}</title></path>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{h - B + 14}" text-anchor="middle">{html.escape(str(c))}</text>')
        out.append(f'<text class="lbl" x="{x + bw / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{fmt.format(v)}</text>')
    out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    if x_label:
        out.append(f'<text x="{(L + w - Rr) / 2:.0f}" y="{h - 2}" text-anchor="middle">{html.escape(x_label)}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(out) + "</svg>"


def tt(ts: pd.Timestamp, entry: pd.Timestamp) -> float:
    return DIX[ts.date()] - DIX[entry.date()] + (ts.hour * 60 + ts.minute - 570) / 390


def agent_section() -> str:
    tf = pd.read_csv(R / "analysis" / "trade_factors.csv")
    paths = pd.read_parquet(R / "analysis" / "agent_trade_paths.parquet")
    paths["ts_et"] = pd.to_datetime(paths["ts_et"])
    cards = []
    for _, r in tf.sort_values("entry_ts").iterrows():
        entry = pd.Timestamp(r["entry_ts"])
        b = r["seen_in"].split()[0].split("/")
        trades = pd.read_csv(R / "agent" / b[0] / b[1] / "trades.csv")
        tr = trades[(trades["ticker"] == r["ticker"])
                    & (pd.to_datetime(trades["entry_ts"]).dt.floor("s") == entry.floor("s"))]
        if tr.empty:
            continue
        sym = tr["raw_symbol"].iloc[0]
        p = paths[paths["key"] == f"{sym}|{entry}"]
        if p.empty:
            continue
        p = p.set_index("ts_et").resample("5min").last().dropna(subset=["bid"]).reset_index()
        cost = float(r["option_cost"])
        pts = [(tt(t, entry), (b_ / cost - 1) * 100) for t, b_ in zip(p["ts_et"], p["bid"])]
        ex = pd.Timestamp(r["exit_ts"])
        sold = (float(tr["exit_price"].iloc[0]) / cost - 1) * 100
        pk_i = int(np.argmax([y for _, y in pts]))
        markers = [{"x": tt(ex, entry), "y": sold, "label": f"sold {sold:+.0f}%", "color": 2, "dy": 14}]
        if pts[pk_i][1] > 5 and abs(pts[pk_i][0] - tt(ex, entry)) > 0.05:
            markers.append({"x": pts[pk_i][0], "y": pts[pk_i][1], "label": f"peak {pts[pk_i][1]:+.0f}%", "color": 1})
        mode = "copy" if "copy" in r["book"] else "real"
        kind = "call" if str(r["opt_type"]).upper().startswith("C") else "put"
        title = f"{r['ticker']} {kind} {r['strike']:g}"
        sub = (f"{entry:%b %-d} → exp {pd.Timestamp(r['expiry']):%b %-d} · {html.escape(r['book'])} · "
               f"{html.escape(str(r['exit_reason']))}")
        chart = line_chart([{"name": title, "color": 1, "pts": pts}], w=360, h=170, markers=markers,
                           compact=True, xmax=max(x for x, _ in pts))
        cards.append(f'<div class="mini"><h3>{html.escape(title)}</h3><p>{sub}</p>{chart}</div>')
    return '<div class="multiples">' + "".join(cards) + "</div>"


def main():
    lab = json.loads((R / "analysis" / "exit_lab.json").read_text())
    rules = pd.read_csv(R / "analysis" / "exit_rules.csv")
    pat = lab["patterns"]
    sections = []

    legend_trade = ('<div class="legend"><span><i style="background:var(--s1)"></i>option value at the bid, '
                    'vs what was paid</span><span><b style="background:var(--s2)"></b>where the agent sold</span>'
                    '<span><b style="background:var(--s1)"></b>best price reached</span></div>')
    sections.append(("Every agent trade",
                     "Each distinct trade the replay agent made in 2025 (all books), from purchase until expiry. "
                     "The line keeps going after the sale so you can see what happened next. Hover for values.",
                     legend_trade + agent_section()))

    s_all = [{"name": "all trades (median)", "color": 1, "pts": [(p[0], p[2]) for p in pat["path_all"]],
              "band": [(p[0], p[1], p[3]) for p in pat["path_all"]]}]
    sections.append(("The typical path",
                     "Median value of the 609 agent-tradable options (spread ≤ 20%, delta ≥ 0.35) over time, with "
                     "the middle half shaded. The bid starts below the price paid because of the spread.",
                     line_chart(s_all, xmax=9)))
    s_two = [{"name": "reached +50% at some point", "color": 1, "pts": [(p[0], p[2]) for p in pat["path_win50"]],
              "band": [(p[0], p[1], p[3]) for p in pat["path_win50"]]},
             {"name": "never reached +50%", "color": 2, "pts": [(p[0], p[2]) for p in pat["path_never50"]],
              "band": [(p[0], p[1], p[3]) for p in pat["path_never50"]]}]
    leg = ('<div class="legend"><span><i style="background:var(--s1)"></i>reached +50% at some point (median, '
           'middle half shaded)</span><span><i style="background:var(--s2)"></i>never reached +50%</span></div>')
    sections.append(("Winners vs the rest", "The winners pull away within the first two or three days; the rest "
                     "bleed steadily toward zero.", leg + line_chart(s_two, xmax=9)))

    pd_ = {int(float(k)): v for k, v in pat["peak_day"].items()}
    tot = sum(pd_.values())
    cats = [str(k) if k < 10 else "10+" for k in sorted(pd_)]
    sections.append(("When the best price came",
                     "Trading day of the hold on which each option hit its best bid (day 0 = the day it was bought).",
                     bar_chart(cats, [pd_[k] / tot * 100 for k in sorted(pd_)], x_label="trading day of the hold")))
    ph = {int(float(k)): v for k, v in pat["peak_halfhour"].items()}
    tot = sum(ph.values())
    labels = {i: f"{(570 + 30 * i) // 60 % 12 or 12}:{(570 + 30 * i) % 60:02d}" for i in range(14)}
    ks = [k for k in sorted(ph) if 0 <= k <= 12]
    sections.append(("Time of day of the best price", "Half-hour in which the best bid happened (ET).",
                     bar_chart([labels[k] for k in ks], [ph[k] / tot * 100 for k in ks], x_label="time of day (ET)")))

    rows = []
    for x, a in pat["after_touch"].items():
        rows.append([f"+{x}%", f"{a['reached']} ({a['share_of_trades']:.0f}%)",
                     f"{a['median_days_to_reach']:.1f}" if a["median_days_to_reach"] is not None else "n/a",
                     f"{a['went_on_to_double_target']:.0f}%", f"{a['ended_below_target']:.0f}%",
                     f"{a['ended_below_zero']:.0f}%", f"{a['median_end']:+.0f}%"])
    sections.append(("What happened after an option first got to +25%, +50%, +100%",
                     "Agent-tradable options only. 'Ended' = value at the bid at the Thursday exit if simply held.",
                     table(["Level", "Options that got there", "Median days to get there", "Went on to double it",
                            "Ended below that level", "Ended below zero", "Median end"], rows)))

    # rules
    t = rules.copy()
    base = t[t["rule"] == "hold to the Thursday exit"].iloc[0]
    v3 = t[(t["tp"].isna()) & (t["sl"] == -75) & (t["trail"] == "50/20") & (t["time_day"].isna())].iloc[0]
    best = t.sort_values("tradable_worst_half_mean", ascending=False).head(12)
    rows = []
    for _, r in pd.concat([pd.DataFrame([v3, base]), best]).drop_duplicates("rule").iterrows():
        tag = " (live V3)" if r["rule"] == v3["rule"] else ""
        rows.append([html.escape(r["rule"]) + tag, f"{r['tradable_H1_mean']:+.1f}%", f"{r['tradable_H2_mean']:+.1f}%",
                     f"{r['tradable_H1_win_rate']:.0f}% / {r['tradable_H2_win_rate']:.0f}%",
                     f"{r['tradable_H1_avg_win']:+.0f}% / {r['tradable_H1_avg_loss']:+.0f}%",
                     f"{r['all_H1_mean']:+.1f}% / {r['all_H2_mean']:+.1f}%"])
    ex = lab["v3_exact_1min"]
    sections.append(("Sell rules compared",
                     f"{len(t):,} rule combinations replayed on every option. Ranked by the WORSE of the two halves, "
                     "so a rule has to work in both. Average return per trade at the bid, agent-tradable options; "
                     f"last column all 1,518. The live V3 rules checked minute-by-minute: "
                     f"{ex['tradable_H1']['mean']:+.1f}% / {ex['tradable_H2']['mean']:+.1f}% (5-minute replay "
                     f"{v3['tradable_H1_mean']:+.1f}% / {v3['tradable_H2_mean']:+.1f}%).",
                     table(["Rule", "Jan-Jun avg", "Jul-Dec avg", "Win rate (H1 / H2)", "Avg win / avg loss (H1)",
                            "All options (H1 / H2)"], rows, cls="corr")))
    body = "".join(
        f'<section><h2>{html.escape(h)}</h2>{f"<p class=sub>{html.escape(s)}</p>" if s else ""}{tb}</section>'
        for h, s, tb in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sell Strategy Lab</title>
<style>{CSS}{VIZ_CSS}</style></head><body class="viz-root"><main>
<h1>When should the agent sell?</h1>
<p class="lead">2025 replay, real OPRA option quotes. Bought at the ask at 9:46, sold at the bid.</p>
{body}
</main>{HOVER_JS}</body></html>"""
    out = R / "report" / "sell_strategy.html"
    out.write_text(page, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
