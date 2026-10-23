# Five-day direction test: outcome (2026-10-23)

**Verdict: no candidate passed.** Pre-registered in `PREREGISTRATION.md`
(commit 664a8ea) before any test-year accuracy was computed. Numbers:
`verdict.json`. Code: `replay/fiveday_test.py`.

Universe: 717 weeks (as-of 2013-01-04 to 2026-09-25), 357,857 S&P 500
stock-weeks; stocks rose over the 5-day window 54.1% of the time.

| Candidate | Accuracy of top-10 calls | 90% range | Over base rate | Worst year | "Up" calls / right | "Down" calls right | Verdict |
|---|---|---|---|---|---|---|---|
| K1 trend20 | 49.9% | 48.5-51.2% | -0.1 pts | 2015, 44.5% | 62% / 51.0% | 48.0% | fail |
| K2 trend60 | 49.7% | 48.2-51.2% | -0.7 pts | 2022, 43.8% | 71% / 51.4% | 45.5% | fail |
| K3 reversal5 | 50.2% | 48.7-51.6% | -0.1 pts | 2019, 44.8% | 44% / 52.1% | 48.6% | fail |
| K4 regime_beta | 50.8% | 48.9-52.7% | +0.3 pts | 2014, 43.5% | 68% / 52.7% | 46.8% | fail |
| K5 gbm (all 23 features) | 53.3% | 51.6-55.0% | +2.3 pts | 2021, 47.9% | 66% / 55.8% | 48.7% | fail |

Bar: pooled >= 60%, every year >= 55%, 90% low end >= 57%, >= 5 points over the
week's base rate. Every candidate failed all four.

- Trend, long trend, reversal and the agent's own SPY-regime rule are coin flips
  on next-week direction.
- The gradient-boosted model has a small, real edge (low end of its range is
  51.6%, +2.3 points over the base rate), concentrated in its "up" calls; its
  "down" calls are below 50%. At 53.3%, the oracle mapping (5-day direction,
  random errors) gives about -1.8% per V3 trade.
- Year by year, K5 ranged from 47.9% (2021) to 60.2% (2025): one good year is not
  a signal.
- Checked: K1 and K3 recomputed from the raw daily table with separate code
  matched exactly (49.86%, 50.15%; same worst years). One stock-week's return and
  20-day feature verified by hand.

Implication: no price-based signal, simple or combined, calls next-week
direction well enough for buying options with V3. The directional-options
approach stops here unless a different kind of information (option-implied,
news) is tested.
