# Out-of-sample spread test: outcome (2026-10-22)

**Verdict: rejected.** The pre-registered rule (`PREREGISTRATION.md`) failed in
both fresh periods. Page: `results/report/oos_spread_test.html`; numbers:
`verdict.json`; every contract: `oos_trades.csv`; weekly sample: `universe.csv`.

| Period | Qualifying trades | Average V3 return | 90% range | 1st / 2nd half | Without best 3% | Without best month | Verdict |
|---|---|---|---|---|---|---|---|
| 2025 (discovery) | 86 | +10.8% | +0.0 to +19.5 | +7.4 / +14.8 | | | (not a test) |
| 2024 | 2,066 | +1.4% | -1.2 to +4.0 | +1.0 / +1.9 | -5.0% | +0.8% | fail (checks 1, 3) |
| Jan-Sep 2026 | 642 | +0.1% | -4.2 to +4.5 | -0.1 / +0.3 | -5.0% | -1.7% | fail (all four) |

- Tight-spread options broke even on new data; the 2025 +10.8% did not
  replicate. Options with 5-20% spreads in the same sample lost -11.3% (2024)
  and -7.2% (2026), so tight spreads are better than wide, but not profitable.
- Calls +7.1% / +4.9%, puts -5.0% / -5.1% (2024 / 2026): the market's drift
  during the holds, not part of the rule.
- Leaving out any one ticker keeps the 2024 average between +1.0% and +1.8%.
  The largest winners are real earnings and news gaps (INTC Aug 2024, PANW Feb
  2024, AVGO Jun 2024, WFC Nov 2024).

## Caveat on 2026 quotes

Entry spreads in the 2026 data are much wider than in 2024 for the same names
(median dollar spread about $0.20 in 2024, $0.55 in the 2025 calm-test sample,
$1.20 in 2026; e.g. JPM 3% in 2024 vs 23% in 2026), so far fewer 2026 options
qualified (642 vs 2,066) and none from the random group. Not checked whether this
is the real market or a quirk of the OPRA consolidated quotes. 2024's quotes look
normal and fail on their own, so the verdict does not depend on 2026.

## Run notes

- Cost quote run 39853163498 (free), real run 39857114047, shard 4 rerun
  39875504214.
- Shard 4 of the real run measured its 5 weeks and saved its data cache, but the
  upload of its results failed (ECONNRESET). It was re-measured from that cache
  with a near-zero budget, so nothing could be bought ($0.00 spent); its original
  purchase of $1.300914 (from its cache ledger, $9.832335 - $8.531421) was added
  to `results/databento_ledger.json` by hand as `oos_shard4_upload_lost`.
- Databento spend for this test: $23.81. Project total $32.34 of $40.
