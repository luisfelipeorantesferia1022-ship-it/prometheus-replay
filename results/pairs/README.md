# Twin-stock pairs test: outcome (2026-10-23)

**Verdict: neither variant passed.** Pre-registered in `PREREGISTRATION.md`
(commit 6d45786) before any test-period trade was simulated. Numbers:
`verdict.json`. Code: `replay/pairs_test.py`. Entries 2013-01-02 to 2026-10-08,
stocks only, net of 0.15% round-trip cost and short borrow.

| Variant | Trades | Net per trade | Before costs | t (by week) | 2013-2019 | 2020-2026 | Without best 3% | Verdict |
|---|---|---|---|---|---|---|---|---|
| P1: 20 economic twins | 1,183 | -0.05% | +0.11% | -0.75 | -0.18% | +0.08% | -0.24% | fail (checks 2, 3, 4) |
| P2: newly tight pairs | 513 | -0.34% | -0.18% | -1.80 | -0.37% | -0.31% | -0.58% | fail (checks 2, 3, 4) |

- **P1:** the twins do come back together sometimes (trades that converged made
  +1.98% on average), but most trades hit the 20-session limit (-0.94%) or the
  stop (-2.48%) first. Before costs the edge is +0.11% per trade, smaller than
  the 0.15% cost, the same pattern as the quant program's short-horizon ideas.
  Win rate 51.8%, average hold 15.5 sessions.
- **Luis's six pairs**, net per trade: WMT/TGT -0.53% (worst of the 20), UPS/FDX
  -0.10%, KO/PEP -0.08%, HD/LOW +0.06%, V/MA +0.16%, NKE/UAA +0.30% (40 trades,
  while both were in the index). With 40-70 trades each, none is distinguishable
  from zero, and choosing the best ones after the fact would be data-mining.
- **P2:** pairs that recently became tightly linked lose money even before costs
  (-0.18% per trade). When a fresh link breaks, it tends to keep breaking.
  Examples picked: AAL/DAL (2015), FIS/GPN (2020), NVDA/NXPI and AMD/KLAC (2023),
  VST/CEG and GS/AXP (2025). Some formations found no newly tight pair at all.
- **Checked:** KO/PEP re-implemented separately from the module matched exactly
  (62 trades, -0.082%); the mean and clustered t recomputed from the trade file
  matched.

Implication: buying the laggard of a twin pair and shorting the leader does not
beat trading costs on S&P 500 stocks since 2013, whether the pairs are chosen by
economics or by a recent tight link. This agrees with the quant program's H2
and with the published decline of pairs trading after 2002.
