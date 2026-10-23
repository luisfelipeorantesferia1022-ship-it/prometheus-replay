# Pairs search then holdout: outcome (2026-10-23)

**Verdict: the chosen setting did not hold up.** Plan locked in
`PREREGISTRATION.md` (commit c938827) before any return was computed; the
selection was committed (daeb7d0, `selection.json`) before the holdout ran
(43b2bae, `holdout.json`). Page: `results/report/pairs_search.html`.
Code: `replay/pairs_search.py`, `replay/pairs_search_report.py`.

## What was done

960 settings of a pairs-trading rule (6 z-score lookbacks x 5 entry gaps x 2 exit
targets x 2 stop rules x 4 maximum holds x 2 ways of matching pairs), with pairs
formed point in time every January and July from the 300 most traded S&P 500
members (same industry, 20 disjoint pairs), stocks only, after 0.15% round-trip
cost and short borrow. Searched on 1999-2012, one setting picked by the plateau
rule, run once on 2013-2026.

## Results

| | Search 1999-2012 | Holdout 2013-2026 |
|---|---|---|
| Selected: distance pairs, 60-day z, enter at 1.0, exit at 0, no stop, max 60 days | Sharpe +0.64, monthly t 2.7, 2,103 trades, +1.6%/yr | **Sharpe -0.25, t -0.96, 1,880 trades, -0.4%/yr** |
| Raw best in the search (same, with a stop) | Sharpe +0.73 | Sharpe -0.24 |
| Share of the 960 settings with a positive Sharpe | 78% | 21% |
| Walk-forward (re-pick every January) | | Sharpe -0.37 |
| Selected rule on random unrelated pairs (median of 200) | | Sharpe -0.21 |

Pre-registered holdout checks: trades >= 100 pass; monthly mean > 0 with t >= 2
**fail** (-0.035%/month, t -0.96); both halves positive **fail** (-0.060% /
-0.008% a month); positive without the best 3% of trades **fail** (-0.21% per
trade); no pair above 25% of P&L pass (largest 2%).

Luck checks, measured on the search period alone before the holdout:
- White's Reality Check, best of 960 above zero: p = 0.295 (not distinguishable
  from luck).
- Deflated Sharpe of the selected setting: 0.30 (needs 0.95; hurdle Sharpe +0.78).
- Same search on random unrelated pairs, 200 times: the real best Sharpe (+0.73)
  was beaten or matched 24% of the time (median random best +0.66, 95th pct +0.90);
  the real best plateau (+0.57) 17% of the time.
- At twice the cost the selected setting's search Sharpe falls to +0.19 and a
  different setting is picked.

All four warned in advance; the holdout confirmed it.

## Other findings

- The search ranking carried a little information (Spearman +0.20 between search
  and holdout Sharpe across the 920 eligible settings), mainly about what to
  avoid: 20-day lookbacks, 5-day holds and 1.0 entry gaps were worst in both
  periods. The search's top tenth averaged a holdout Sharpe of -0.25.
- Walk-forward chased past winners: every setting it picked did worse in the
  years it was picked than in the other holdout years.
- The holdout's own best settings (correlation pairs, 120-250-day lookbacks,
  entry 2-2.5, 60-day holds, holdout Sharpe +0.58 to +0.67) were middling in the
  search and earned much of it in 2020. Picking them now would be the same
  mistake; they would need a fresh test.
- Holdout trades: 1,420 closed when the gap closed (+0.83% average, 81% winners),
  324 were still open at the six-month cutoff (-1.70%), 132 hit the 60-day limit
  (-5.63%). 18.5% of "gap closed" trades still lost money, because a rolling
  z-score can return to zero by its average catching up rather than the prices
  converging.
- Search profits came mostly from 1999 (+8.3%) and 2008-09 (+6.0%, +5.4%);
  2010-2012 were already negative.

## Verification

- `tests/pairs_search_check.py`: the simulator equals an independently written
  reference on synthetic data for all 480 rule settings (126,686 trades, every
  exit type), z-scores equal pandas rolling, grid and pair rules checked.
- `tests/pairs_search_verify.py`: an independent re-implementation of the whole
  holdout for the selected setting (universe, pairs, z-scores, trades, daily
  portfolio) matched all 1,880 trades exactly (net difference 1e-16), the Sharpe
  (-0.2456), monthly t (-0.961) and both halves; one trade (ADP/PAYX, July to
  September 2019) recomputed by hand from the raw rows matched.

Per-trade detail and daily return arrays are kept off this public repo
(licensed price data).
