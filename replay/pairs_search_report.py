"""
Results page for the pairs search-then-holdout test (results/pairs_search/).

    python -m replay.pairs_search_report --scratch DIR   # results/report/pairs_search.html
DIR is the scratch folder the test wrote its daily return arrays to; the page's
monthly series are saved to results/pairs_search/monthly_returns.csv.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .exit_report import VIZ_CSS, nice_ticks
from .report_tables import CSS, table

R = config.RESULTS_DIR
D = R / "pairs_search"

PAGE_JS = """
<div class="tip" id="tip"></div>
<script>
(function(){
  const tip = document.getElementById('tip');
  function show(ev, rows) {
    tip.textContent = '';
    rows.forEach((r, i) => {
      const row = document.createElement('div');
      if (r[0] !== null) { const v = document.createElement('span'); v.className = 'v'; v.textContent = r[0]; row.appendChild(v); }
      if (r[1]) { const k = document.createElement('span'); k.className = 'k'; k.textContent = (r[0] !== null ? '  ' : '') + r[1]; row.appendChild(k); }
      tip.appendChild(row);
    });
    tip.style.display = 'block';
    const x = ev.clientX !== undefined ? ev.clientX : 20, y = ev.clientY !== undefined ? ev.clientY : 20;
    tip.style.left = Math.min(x + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (y + 12) + 'px';
  }
  function hide() { tip.style.display = 'none'; }
  // bars and labelled marks
  document.querySelectorAll('[data-tip]').forEach(el => {
    let rows; try { rows = JSON.parse(el.getAttribute('data-tip')); } catch(e) { return; }
    el.addEventListener('pointermove', ev => show(ev, rows));
    el.addEventListener('pointerleave', hide);
    el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); show({clientX: r.right, clientY: r.top}, rows); });
    el.addEventListener('blur', hide);
  });
  const ns = 'http://www.w3.org/2000/svg';
  // scatter: nearest point within 24px
  document.querySelectorAll('svg[data-scatter]').forEach(svg => {
    let d; try { d = JSON.parse(svg.getAttribute('data-scatter')); } catch(e) { return; }
    const ring = document.createElementNS(ns, 'circle');
    ring.setAttribute('r', '7'); ring.setAttribute('fill', 'none'); ring.setAttribute('stroke', 'var(--ink)');
    ring.setAttribute('stroke-width', '1.5'); ring.style.display = 'none'; svg.appendChild(ring);
    svg.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
      const px = (ev.clientX - r.left) * vb.width / r.width, py = (ev.clientY - r.top) * vb.height / r.height;
      let best = null, bd = 24 * vb.width / r.width;
      d.pts.forEach(p => { const dd = Math.hypot(p[0] - px, p[1] - py); if (dd < bd) { bd = dd; best = p; } });
      if (!best) { hide(); ring.style.display = 'none'; return; }
      ring.setAttribute('cx', best[0]); ring.setAttribute('cy', best[1]); ring.style.display = '';
      show(ev, best[2]);
    });
    svg.addEventListener('pointerleave', () => { hide(); ring.style.display = 'none'; });
  });
  // line: crosshair snapping to the nearest month
  document.querySelectorAll('svg[data-line]').forEach(svg => {
    let d; try { d = JSON.parse(svg.getAttribute('data-line')); } catch(e) { return; }
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('stroke', 'var(--axis)'); line.setAttribute('stroke-width', '1');
    line.setAttribute('y1', d.top); line.setAttribute('y2', d.bottom); line.style.display = 'none'; svg.appendChild(line);
    svg.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
      const px = (ev.clientX - r.left) * vb.width / r.width;
      if (px < d.left || px > d.right) { hide(); line.style.display = 'none'; return; }
      let bi = 0, bd = 1e9;
      d.xs.forEach((x, i) => { const dd = Math.abs(x - px); if (dd < bd) { bd = dd; bi = i; } });
      line.setAttribute('x1', d.xs[bi]); line.setAttribute('x2', d.xs[bi]); line.style.display = '';
      const rows = [[null, d.labels[bi]]];
      d.series.forEach(s => { const v = s.v[bi]; if (v !== null) rows.push([(v > 0 ? '+' : '') + v.toFixed(1) + '%', s.name]); });
      show(ev, rows);
    });
    svg.addEventListener('pointerleave', () => { hide(); line.style.display = 'none'; });
  });
})();
</script>"""

EXTRA_CSS = """
.viz-root { --s3:#1baf7a; --ref:#52514e; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { --s3:#199e70; --ref:#c3c2b7; } }
:root[data-theme="dark"] .viz-root { --s3:#199e70; --ref:#c3c2b7; }
.verdict { display:flex; gap:28px; flex-wrap:wrap; margin:4px 0 14px; }
.verdict div { min-width:150px; } .verdict .lab { color:var(--muted); font-size:13px; }
.verdict .num { font-size:30px; font-weight:600; line-height:1.2; }
.pill { display:inline-block; padding:1px 9px; border-radius:999px; font-size:12px; font-weight:600;
        border:1px solid var(--line); }
details { margin:0 0 12px; } summary { cursor:pointer; color:var(--muted); font-size:13px; margin:0 0 8px; }
"""


def cover_ticks(lo, hi, n=6):
    """Clean ticks whose first and last values enclose [lo, hi]."""
    span = max(hi - lo, 1e-9)
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    a, b = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    return np.round(np.arange(a, b + step / 2, step), 10)


def esc(x) -> str:
    return html.escape(str(x))


def describe(r) -> str:
    lb = r["lookback"]
    lb = "static z" if str(lb) == "static" else f"{lb}-day z"
    stop = "stop at entry+2" if r["stop"] == "entry+2" else "no stop"
    tgt = "exit at 0" if float(r["exit_within"]) == 0 else "exit within 0.5"
    meth = "distance" if str(r["method"]).startswith("S1") else "correlation"
    return f"{meth} pairs, {lb}, enter at {float(r['entry_z']):g}, {tgt}, {stop}, max {int(r['max_hold'])} days"


def histogram(vals, marks, refs, w=720, h=300, bw_=0.1):
    """vals: Sharpe ratios; marks: [(x, label)] settings (orange lines); refs: [(x, label)] luck lines (grey)."""
    L, Rr, T, B = 40, 16, 14, 36
    lo = math.floor(min(min(vals), *(m[0] for m in marks + refs)) / bw_) * bw_
    hi = math.ceil(max(max(vals), *(m[0] for m in marks + refs)) / bw_) * bw_ + 1e-9
    edges = np.arange(lo, hi + bw_, bw_)
    cnt, _ = np.histogram(vals, edges)
    yt = cover_ticks(0, cnt.max() * 1.55, 4)
    ymax = yt[-1]
    X = lambda x: L + (x - edges[0]) / (edges[-1] - edges[0]) * (w - L - Rr)
    Y = lambda y: T + (ymax - y) / ymax * (h - T - B)
    out = []
    for t in yt:
        out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:.0f}</text>')
    slot = X(edges[1]) - X(edges[0])
    for i, c in enumerate(cnt):
        if not c:
            continue
        x0, x1 = X(edges[i]) + 1, X(edges[i + 1]) - 1
        y = Y(c)
        r = min(4, (Y(0) - y) / 2, (x1 - x0) / 2)
        tipr = json.dumps([[f"{c} settings", f"Sharpe {edges[i]:+.1f} to {edges[i + 1]:+.1f}"]])
        out.append(f'<path tabindex="0" data-tip=\'{esc(tipr)}\' d="M{x0:.1f},{Y(0):.1f} V{y + r:.1f} '
                   f'Q{x0:.1f},{y:.1f} {x0 + r:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
                   f'V{Y(0):.1f} Z" fill="var(--s1)"/>')
    out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    for t in nice_ticks(edges[0], edges[-1], 8):
        if edges[0] - 1e-9 <= t <= edges[-1] + 1e-9:
            out.append(f'<text x="{X(t):.1f}" y="{h - B + 15}" text-anchor="middle">{t:+.1f}'
                       .replace("+0.0", "0") + '</text>')
    out.append(f'<text x="{(L + w - Rr) / 2:.0f}" y="{h - 4}" text-anchor="middle">annual Sharpe ratio, 1999-2012, after costs</text>')
    items = sorted([(x, lab, "var(--ref)") for x, lab in refs] + [(x, lab, "var(--s2)") for x, lab in marks],
                   key=lambda q: -q[0])
    for row, (x, lab, col) in enumerate(items):       # rightmost on the top row; lines start below their label
        yl = T + 10 + 15 * row
        out.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{yl + 4}" y2="{Y(0):.1f}" stroke="{col}" stroke-width="1.5"/>')
        out.append(f'<text class="lbl" x="{X(x) - 4:.1f}" y="{yl:.1f}" text-anchor="end">{esc(lab)} {x:+.2f}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Histogram of search Sharpe ratios">' + "".join(out) + "</svg>"


def scatter(df, sel, raw, w=720, h=400):
    L, Rr, T, B = 46, 16, 14, 40
    xs, ys = df["search_sharpe"].to_numpy(), df["holdout_sharpe"].to_numpy()
    xt = cover_ticks(min(xs.min(), 0), max(xs.max(), 0), 6)
    yt = cover_ticks(min(ys.min(), 0), max(ys.max(), 0), 6)
    X = lambda x: L + (x - xt[0]) / (xt[-1] - xt[0]) * (w - L - Rr)
    Y = lambda y: T + (yt[-1] - y) / (yt[-1] - yt[0]) * (h - T - B)
    out = []
    for t in yt:
        out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:+.1f}</text>'.replace("+0.0", "0"))
    for t in xt:
        out.append(f'<text x="{X(t):.1f}" y="{h - B + 15}" text-anchor="middle">{t:+.1f}</text>'.replace("+0.0", "0"))
    out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    out.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{T}" y2="{h - B}" stroke="var(--axis)" stroke-width="1"/>')
    out.append(f'<text x="{(L + w - Rr) / 2:.0f}" y="{h - 4}" text-anchor="middle">Sharpe in the search, 1999-2012</text>')
    out.append(f'<text x="12" y="{(T + h - B) / 2:.0f}" text-anchor="middle" transform="rotate(-90 12 {(T + h - B) / 2:.0f})">'
               f'Sharpe in the holdout, 2013-2026</text>')
    pts = []
    for meth, col in (("S1_distance", 1), ("S2_correlation", 2)):
        sub = df[df["method"] == meth]
        for c, r in sub.iterrows():
            cx, cy = X(r["search_sharpe"]), Y(r["holdout_sharpe"])
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="var(--s{col})" fill-opacity=".55"/>')
            pts.append([round(cx, 1), round(cy, 1), [[f"{r['holdout_sharpe']:+.2f}", "holdout Sharpe"],
                                                     [f"{r['search_sharpe']:+.2f}", "search Sharpe"],
                                                     [None, describe(r)]]])
    for c, lab, dy in ((raw, "raw best in the search", -10), (sel, "selected (best plateau)", 16)):
        r = df.loc[c]
        cx, cy = X(r["search_sharpe"]), Y(r["holdout_sharpe"])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="var(--ink)" stroke="var(--surf)" stroke-width="2"/>')
        anchor = "end" if cx > w * 0.6 else "start"
        out.append(f'<text class="lbl" x="{cx + (-9 if anchor == "end" else 9):.1f}" y="{cy + dy:.1f}" '
                   f'text-anchor="{anchor}">{esc(lab)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Search versus holdout Sharpe for 960 settings" '
            f'data-scatter=\'{esc(json.dumps({"pts": pts}))}\'>' + "".join(out) + "</svg>")


def line_chart(labels, series, split=None, w=720, h=280, ylab="cumulative return after costs"):
    """series: [{name, color, v: [float|None]}] on a shared monthly x."""
    L, Rr, T, B = 46, 110, 12, 34
    n = len(labels)
    vals = [v for s in series for v in s["v"] if v is not None]
    yt = cover_ticks(min(vals + [0]), max(vals + [0]), 5)
    X = lambda i: L + i / max(n - 1, 1) * (w - L - Rr)
    Y = lambda y: T + (yt[-1] - y) / (yt[-1] - yt[0]) * (h - T - B)
    out = []
    for t in yt:
        out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{L - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{t:+.0f}%</text>'.replace("+0%", "0"))
    out.append(f'<line x1="{L}" x2="{w - Rr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    years = [int(l[:4]) for l in labels]
    step = 2 if years[-1] - years[0] > 16 else 1
    for i, l in enumerate(labels):
        if l.endswith("-01") and (years[i] - years[0]) % step == 0:
            out.append(f'<text x="{X(i):.1f}" y="{h - B + 15}" text-anchor="middle">{years[i]}</text>')
    out.append(f'<text x="{(L + w - Rr) / 2:.0f}" y="{h - 3}" text-anchor="middle">{esc(ylab)}</text>')
    if split is not None:
        out.append(f'<line x1="{X(split):.1f}" x2="{X(split):.1f}" y1="{T}" y2="{h - B}" stroke="var(--ref)" stroke-width="1"/>')
        out.append(f'<text class="lbl" x="{X(split) - 6:.1f}" y="{h - B - 6}" text-anchor="end">search, 1999-2012</text>')
        out.append(f'<text class="lbl" x="{X(split) + 6:.1f}" y="{h - B - 6}">holdout, 2013-2026</text>')
    ends = []
    for s in series:
        segs, cur = [], []
        for i, v in enumerate(s["v"]):
            if v is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{X(i):.1f},{Y(v):.1f}")
        if cur:
            segs.append(cur)
        for sg in segs:
            out.append(f'<polyline points="{" ".join(sg)}" fill="none" stroke="var(--s{s["color"]})" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')
        last = max(i for i, v in enumerate(s["v"]) if v is not None)
        ends.append([Y(s["v"][last]), s, last])
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):                      # keep end labels apart, with leader lines
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 14)
    for yl, s, last in ends:
        yv = Y(s["v"][last])
        out.append(f'<circle cx="{X(last):.1f}" cy="{yv:.1f}" r="4" fill="var(--s{s["color"]})" stroke="var(--surf)" stroke-width="2"/>')
        if abs(yl - yv) > 1:
            out.append(f'<line x1="{X(last) + 5:.1f}" y1="{yv:.1f}" x2="{X(last) + 12:.1f}" y2="{yl:.1f}" stroke="var(--axis)" stroke-width="1"/>')
        out.append(f'<text class="lbl" x="{X(last) + 14:.1f}" y="{yl + 4:.1f}">{esc(s["name"])} {s["v"][last]:+.0f}%</text>')
    hover = {"left": L, "right": w - Rr, "top": T, "bottom": h - B, "xs": [round(X(i), 1) for i in range(n)],
             "labels": labels, "series": [{"name": s["name"], "v": [None if v is None else round(v, 2) for v in s["v"]]}
                                          for s in series]}
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Cumulative return" '
            f'data-line=\'{esc(json.dumps(hover))}\'>' + "".join(out) + "</svg>")


def cum_pct(r: pd.Series) -> list:
    return list(((1 + r).cumprod() - 1) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    sc = Path(a.scratch)
    sel = json.loads((D / "selection.json").read_text())
    ho = json.loads((D / "holdout.json").read_text())
    res = pd.read_csv(D / "search_results.csv", index_col="config")
    allc = pd.read_csv(D / "holdout_all_settings.csv", index_col="config")
    s_id, r_id = sel["selected"]["config"], sel["raw_best"]["config"]

    # monthly series for the charts (saved so the page can be rebuilt without the scratch arrays)
    from .pairs_search import Panel  # noqa: F401  (calendar comes from the arrays' saved offsets)
    sd, hd = np.load(sc / "search_daily.npz"), np.load(sc / "holdout_daily.npz")
    cal = pd.read_parquet(sc / "calendar.parquet")["date"]
    ds = pd.DatetimeIndex(cal.iloc[int(sd["day0"]):int(sd["day0"]) + sd["daily"].shape[1]])
    dh = pd.DatetimeIndex(cal.iloc[int(hd["day0"]):int(hd["day0"]) + hd["daily"].shape[1]])
    sel_full = pd.Series(np.concatenate([sd["daily"][s_id], hd["selected"]]), index=ds.append(dh))
    mo = lambda s: (1 + s).groupby(s.index.to_period("M")).prod() - 1
    m_sel = mo(sel_full)
    m_wf = mo(pd.Series(hd["walk_forward"], index=dh))
    m_raw = mo(pd.Series(hd["daily"][r_id], index=dh))
    m_sel_h = mo(pd.Series(hd["selected"], index=dh))
    monthly = pd.DataFrame({"selected": m_sel, "walk_forward": m_wf, "raw_best_holdout": m_raw})
    monthly.index = monthly.index.astype(str)
    monthly.to_csv(D / "monthly_returns.csv", float_format="%.6g")

    sections = []
    s, hs = sel["selected"], ho["summary"]
    crit_names = {"1_trades_ge_100": "At least 100 trades",
                  "2_monthly_mean_pos_and_t_ge_2": "Average monthly return above zero with t >= 2",
                  "3_both_halves_pos": "Positive in 2013-2019 and in 2020-2026",
                  "4_trade_mean_pos_without_top_3pct": "Positive per trade without the best 3% of trades",
                  "5_no_pair_over_25pct": "No pair above 25% of the profit and loss"}
    crit_vals = {"1_trades_ge_100": f"{ho['trades']:,} trades",
                 "2_monthly_mean_pos_and_t_ge_2": f"{hs['mean_monthly_pct']:+.3f}% a month, t {hs['t_monthly']:.2f}",
                 "3_both_halves_pos": " / ".join(f"{v:+.3f}%" for v in ho["halves_mean_monthly_pct"]),
                 "4_trade_mean_pos_without_top_3pct": f"{ho['drop_top']['mean_net_per_trade_pct']:+.3f}% per trade "
                                                      f"without the best {ho['drop_top']['k']}",
                 "5_no_pair_over_25pct": f"largest pair {ho['max_pair_share'] * 100:.0f}% ({esc(ho['top_pair'])})"}
    verdict_txt = "Held up" if ho["verdict"] == "pass" else "Did not hold up"
    head = (f'<div class="verdict"><div><div class="lab">Sharpe in the search, 1999-2012</div>'
            f'<div class="num">{s["sharpe"]:+.2f}</div></div>'
            f'<div><div class="lab">Sharpe in the holdout, 2013-2026</div><div class="num">{hs["sharpe"]:+.2f}</div></div>'
            f'<div><div class="lab">Return a year in the holdout</div><div class="num">{hs["annual_return_pct"]:+.1f}%</div></div>'
            f'<div><div class="lab">Verdict</div><div class="num">{verdict_txt}</div></div></div>')
    rows = [[esc(crit_names[k]), crit_vals[k], "pass" if v else "<b>fail</b>"] for k, v in ho["criteria"].items()]
    sections.append(("The chosen setting on new data",
                     f"Picked from 960 settings on 1999-2012 by the plateau rule: {describe(s)}. "
                     f"Run once on 2013-2026 with pairs re-formed point in time every six months.",
                     head + table(["Pre-registered check", "Result", ""], rows)))

    rn, rc = sel["random_pair_null"], sel["reality_check"]
    vals = res.loc[res["sharpe"].notna(), "sharpe"].to_numpy()
    hist = histogram(vals, marks=[(res.loc[r_id, "sharpe"], "raw best"), (s["sharpe"], "selected")],
                     refs=[(rn["null_best_sharpe_95pct"], "random pairs, best of 960 (95th pct)"),
                           (rn["null_best_sharpe_median"], "random pairs, best of 960 (median)")])
    luck = table(["Check", "Result", "What it means"], [
        ["Settings with a positive Sharpe in the search", f"{sel['share_of_settings_with_positive_sharpe'] * 100:.0f}%",
         f"median Sharpe {sel['median_sharpe']:+.2f}"],
        ["Raw best Sharpe vs the same search on random unrelated pairs",
         f"{rn['real_best_sharpe']:+.2f} vs median {rn['null_best_sharpe_median']:+.2f}, 95th pct "
         f"{rn['null_best_sharpe_95pct']:+.2f}", f"p = {rn['p_best_sharpe']:.3f}"],
        ["Selected plateau vs random pairs", f"{rn['real_best_plateau']:+.2f} vs median "
         f"{rn['null_best_plateau_median']:+.2f}, 95th pct {rn['null_best_plateau_95pct']:+.2f}",
         f"p = {rn['p_best_plateau']:.3f}"],
        ["Reality Check: is the best of 960 above zero beyond luck?", f"p = {rc['p_value']:.3f}",
         "below 0.05 means yes"],
        ["Deflated Sharpe of the selected setting (960 trials)", f"{sel['deflated_sharpe_selected']['dsr']:.3f}",
         f"needs 0.95; hurdle Sharpe {sel['deflated_sharpe_selected']['hurdle_sharpe_annual']:+.2f}"],
        ["Same search at twice the cost", "same setting picked" if sel["selection_at_2x_cost"]["same_as_selected"]
         else f"different setting picked (#{sel['selection_at_2x_cost']['config']})",
         f"selected Sharpe at 2x cost {sel['selection_at_2x_cost']['selected_sharpe_at_2x_cost']:+.2f}"]])
    sections.append(("How good the best of 960 looked by luck",
                     "Every setting's Sharpe ratio in the search. Orange lines: the selected setting and the raw best. "
                     "Grey lines: how good the best of 960 looks when the identical search runs on random pairs of "
                     "unrelated stocks (200 repetitions).",
                     f'<div class="viz-root">{hist}</div>' + luck))

    al = allc[allc["eligible"].astype(bool)]
    sc_svg = scatter(allc.dropna(subset=["search_sharpe", "holdout_sharpe"]), s_id, r_id)
    sa = ho["secondary"]["all_settings"]
    dec_rows = [[str(k), f"{v['search_sharpe']:+.2f}", f"{v['holdout_sharpe']:+.2f}"] for k, v in sa["by_search_decile"].items()]
    sections.append(("Did a good search predict a good holdout?",
                     f"All 960 settings: search Sharpe against holdout Sharpe. Spearman correlation across the "
                     f"{sa['eligible']} eligible settings: {sa['spearman_search_vs_holdout_sharpe']:+.2f}. "
                     f"{sa['share_positive_holdout_sharpe'] * 100:.0f}% of them had a positive holdout Sharpe.",
                     '<div class="viz-root"><div class="legend"><span><b style="background:var(--s1)"></b>distance pairs</span>'
                     '<span><b style="background:var(--s2)"></b>correlation pairs</span>'
                     '<span><b style="background:var(--ink)"></b>the two settings marked</span></div>'
                     f'{sc_svg}</div>'
                     + "<details><summary>Average holdout Sharpe by search decile (1 = worst in the search)</summary>"
                     + table(["Search decile", "Search Sharpe", "Holdout Sharpe"], dec_rows, cls="corr") + "</details>"))

    labels = list(monthly.index)
    split = labels.index(pd.Period("2013-01", "M").strftime("%Y-%m"))
    full_svg = line_chart(labels, [{"name": "selected", "color": 1, "v": cum_pct(monthly["selected"])}], split=split)
    hl = labels[split:]
    wfv = monthly["walk_forward"].iloc[split:]
    hold_svg = line_chart(hl, [{"name": "selected", "color": 1, "v": cum_pct(m_sel_h)},
                               {"name": "walk-forward", "color": 2, "v": cum_pct(wfv)},
                               {"name": "raw best", "color": 3, "v": cum_pct(monthly["raw_best_holdout"].iloc[split:])}])
    wf = ho["secondary"]["walk_forward"]
    rb = ho["secondary"]["raw_best_holdout"]
    rp = ho["secondary"]["selected_on_random_pairs"]
    yr_rows = []
    by_sel = {**{int(k): v for k, v in sel["selected_summary"]["by_year_pct"].items()},
              **{int(k): v for k, v in hs["by_year_pct"].items()}}
    for y in sorted(by_sel):
        yr_rows.append([str(y), f"{by_sel[y]:+.2f}%", "search" if y <= 2012 else "holdout"])
    sections.append(("The chosen setting over time",
                     "Cumulative return after costs on the 20-pair portfolio, January 1999 to October 2026.",
                     f'<div class="viz-root">{full_svg}</div>'
                     + "<details><summary>Return by year</summary>" + table(["Year", "Return", "Period"], yr_rows, cls="corr")
                     + "</details>"))
    sections.append(("Holdout only: the chosen setting, re-tuning every year, and the raw best",
                     "From January 2013. Walk-forward re-picks the setting every January by the same plateau rule "
                     "using everything from 1999 to the previous December.",
                     '<div class="viz-root"><div class="legend"><span><i style="background:var(--s1)"></i>selected</span>'
                     '<span><i style="background:var(--s2)"></i>walk-forward</span>'
                     '<span><i style="background:var(--s3)"></i>raw best</span></div>'
                     f'{hold_svg}</div>'
                     + table(["Version", "Holdout Sharpe", "Return a year", "Max drawdown"], [
                         ["Selected (pre-registered)", f"{hs['sharpe']:+.2f}", f"{hs['annual_return_pct']:+.1f}%",
                          f"{hs['max_drawdown_pct']:.1f}%"],
                         ["Walk-forward", f"{wf['summary']['sharpe']:+.2f}", f"{wf['summary']['annual_return_pct']:+.1f}%",
                          f"{wf['summary']['max_drawdown_pct']:.1f}%"],
                         ["Raw best in the search", f"{rb['summary']['sharpe']:+.2f}",
                          f"{rb['summary']['annual_return_pct']:+.1f}%", f"{rb['summary']['max_drawdown_pct']:.1f}%"],
                         ["Selected rule on random unrelated pairs (median of 200)", f"{rp['random_median']:+.2f}", "", ""]],
                         cls="corr")))
    tr = pd.read_csv(sc / "holdout_trades_selected.csv")
    g = tr.groupby("reason")["net"]
    order = ["converged", "period end", "time", "data end"]
    ex_rows = [[{"converged": "Gap closed (z crossed 0)", "period end": "Still open at the six-month cutoff",
                 "time": "Held the full 60 days", "data end": "A stock stopped trading"}[k], f"{g.size()[k]:,}",
                f"{g.mean()[k] * 100:+.2f}%", f"{(tr.loc[tr['reason'] == k, 'net'] > 0).mean() * 100:.0f}%"]
               for k in order if k in g.size()]
    ex_rows.append(["All trades", f"{len(tr):,}", f"{tr['net'].mean() * 100:+.2f}%", f"{(tr['net'] > 0).mean() * 100:.0f}%"])
    sections.append(("How the holdout trades ended",
                     "Net return per trade on the pair's notional, after costs.",
                     table(["Exit", "Trades", "Average", "Winners"], ex_rows, cls="corr")))
    best_h = al.sort_values("holdout_sharpe", ascending=False).head(5)
    bh_rows = [[describe(r), f"{r['search_sharpe']:+.2f}", f"{r['holdout_sharpe']:+.2f}"] for _, r in best_h.iterrows()]
    notes = f"""<ul>
<li>The search found a setting that made money in 1999-2012 (Sharpe {s['sharpe']:+.2f}, monthly t
{sel['selected_summary']['t_monthly']:.1f}) and it lost money in 2013-2026 (Sharpe {hs['sharpe']:+.2f}). The luck checks said so
in advance: the Reality Check could not tell the best of 960 from chance (p {rc['p_value']:.2f}), the Deflated Sharpe was
{sel['deflated_sharpe_selected']['dsr']:.2f} against the 0.95 needed, and random unrelated pairs produced an equally good
best result {rn['p_best_sharpe'] * 100:.0f}% of the time.</li>
<li>Pairs trading paid a little for most settings in 1999-2012, mostly in 1999 and the 2008-09 crisis, then stopped:
{sel['share_of_settings_with_positive_sharpe'] * 100:.0f}% of settings were positive in the search, {sa['share_positive_holdout_sharpe'] * 100:.0f}% in the
holdout. This matches the published decline of pairs profits after 2002.</li>
<li>The search ranking carried a little information (rank correlation {sa['spearman_search_vs_holdout_sharpe']:+.2f}), mostly about what to
avoid (20-day lookbacks, 5-day holds, entering at a 1.0 gap), but even the search's top tenth lost money in the holdout.</li>
<li>Re-tuning every January did worse (Sharpe {wf['summary']['sharpe']:+.2f}): each setting it picked did worse in the years it was
picked than in the other years. That is chasing last period's winner.</li>
<li>The selected rule did no better on its same-industry pairs than on random unrelated pairs (median
{rp['random_median']:+.2f}), so the twin relationship added nothing in the holdout.</li>
<li>The holdout's own best settings (table below) were middling in the search, and much of their gain came from 2020.
Picking them now would repeat the mistake this test was built to catch; they would need a fresh test on data from 2027 on,
or paper trading.</li>
</ul>""" + table(["Best five settings in the holdout (hindsight)", "Search Sharpe", "Holdout Sharpe"], bh_rows, cls="corr")
    sections.append(("What it means", "", notes))
    nb_rows = [[describe(r), f"{r['sharpe']:+.2f}", str(r["trades"])] for r in [s] + sel["selected_neighbours"]]
    sections.append(("The selected setting and its neighbours in the search",
                     "The plateau score is the average of these Sharpe ratios.",
                     table(["Setting", "Search Sharpe", "Trades"], nb_rows, cls="corr")))
    body = "".join(f'<section><h2>{esc(h)}</h2>{f"<p class=sub>{esc(t)}</p>" if t else ""}{tb}</section>'
                   for h, t, tb in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pairs Search Test</title>
<style>{CSS}{VIZ_CSS}{EXTRA_CSS}</style></head><body><main>
<h1>Search many settings, then test once</h1>
<p class="lead">Pairs trading on S&amp;P 500 stocks: 960 settings searched on 1999-2012, one picked by a rule fixed in
advance, tested once on 2013-2026. Pre-registered (results/pairs_search/PREREGISTRATION.md); Sharadar daily prices,
point-in-time index membership, after costs.</p>
{body}
</main>{PAGE_JS}</body></html>"""
    out = R / "report" / "pairs_search.html"
    out.write_text(page, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
