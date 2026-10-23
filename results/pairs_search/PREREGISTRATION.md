# Pairs trading, search then holdout — pre-registration

Written and committed on 2026-10-23, before any return of any setting in either
period was computed. Before this commit only these were run: the synthetic checks
in `tests/pairs_search_check.py` (no market data) and a plumbing run on the real
data that printed the pairs formed at 4 search and 4 holdout formations and the
number of sessions per pair-period, with no returns. Nothing below may change
after results are seen.

Code: `replay/pairs_search.py`. Data: Sharadar daily prices 1997-12-31 to
2026-10-08 (split- and dividend-adjusted closes, opens on the same basis),
point-in-time S&P 500 membership (snapshots from 1998-03-31), and Sharadar's
industry and sector labels, derived from Luis's bulk files into
`quant/data/sharadar/derived/` (licensed: not committed). Stocks only, no options.

## The question

Can trying many settings of a pairs-trading rule, and keeping the one that works,
be told apart from a backtest fooling itself? Method: search only on 1999-2012,
pick the winner by a rule fixed here, measure in advance how good the best of 960
settings looks by luck, then run the chosen setting once on 2013-2026.

## Pairs (point in time)

On the first session of every January and July (formation date F):
- Universe: S&P 500 members as of the session before F, with a complete adjusted
  close for each of the 252 sessions before F, unadjusted price >= $5 on the
  session before F; the 300 with the highest average dollar volume over the 60
  sessions before F.
- Candidate pairs: two stocks with the same Sharadar industry (SIC industry if
  missing), excluding share classes of one company (same normalized name).
- **S1 distance:** smallest sum of squared differences of the two normalized price
  paths (adjusted close / its first value) over the 252 sessions before F.
- **S2 correlation:** highest correlation of daily log returns over those 252
  sessions.
- Each method takes candidates in order and keeps a pair only if neither stock is
  already used, until it has 20 pairs.
- The pairs trade from F to the session before the next formation date (about six
  months). Anything still open is closed at the close of the last session.

## Trading rule and the 960 settings

Spread = ln(A) - ln(B) of adjusted closes; z = (spread - mean) / sd.

| Dimension | Values |
|---|---|
| Lookback for the mean and sd | 20, 40, 60, 120, 250 sessions (rolling, including today), or static: the 252-session formation window |
| Entry gap | |z| >= 1.0, 1.5, 2.0, 2.5, 3.0 |
| Exit target | gap back to 0 (z crosses 0), or back within 0.5 sd |
| Stop | none, or when the gap widens to entry + 2 sd |
| Maximum hold | 5, 10, 20, 60 sessions |
| Pair method | S1 distance, S2 correlation |

6 x 5 x 2 x 2 x 4 x 2 = 960 settings.

Mechanics (all settings): the signal is read at a close and traded at the next
open; buy the cheap stock and short the rich one, equal dollars (each leg half
the pair's slot); no entry if the next open is the period's last session. Exits
(checked at each close, traded at the next open): target reached, stop reached,
or the maximum hold reached (sessions counted from the entry day). At the start
of each period a pair can enter at once; after any exit it can re-enter only after
|z| has been below half the entry gap at a close. If a stock stops trading, the pair closes at its last close. Costs: 0.15%
of the pair's notional per round trip, plus 0.5% a year borrow on the short half.
Portfolio: 20 equal slots (committed capital); daily return = sum of the pairs'
daily P&L / 20, zero for idle slots.

## Search (1999-2012)

- Trading periods from the January 1999 formation through the July 2012
  formation's period (28 periods, sessions 1999-01-04 to 2012-12-31).
- Score per setting: annualized Sharpe ratio of the portfolio's daily net return
  over every session of the period.
- Eligible: at least 300 completed trades in the search period.
- **Selection rule:** the eligible setting with the highest plateau score, the
  average Sharpe of the setting and its one-step neighbours (settings differing
  in exactly one dimension by one step: lookback in the order 20, 40, 60, 120,
  250, static; entry gap +/-0.5; exit target or stop switched; maximum hold one
  value up or down; the other pair method). Ties: higher own Sharpe. This rewards
  a plateau of good settings over a lone spike.

## Measured before the holdout (reported, not part of the verdict)

- The raw best setting (highest own Sharpe, eligible).
- White's Reality Check on all 960 daily series: studentized maximum of
  sqrt(T) x mean / sd against a stationary bootstrap (mean block 20 sessions,
  2,000 resamples, seed 20261023). Tests whether the best of 960 beats zero by
  more than luck.
- Deflated Sharpe Ratio of the selected and raw best settings, with 960 trials
  and the variance of the 960 Sharpe ratios.
- Random-pair null: the whole search repeated 200 times with each pair method
  replaced by 20 random disjoint pairs of stocks from different sectors among the
  same 300 stocks; percentile of the real best Sharpe and best plateau score
  among the 200 random bests.
- The search repeated at twice the cost: does the selection change?

## Holdout (2013-2026): one look at the selected setting

Trading periods from the January 2013 formation through the July 2026
formation's period (sessions 2013-01-02 to 2026-10-08). The holdout stage
refuses to run unless the search output `selection.json` exists; it is
committed before the holdout is run.

**Pass bar (all must hold):**
1. At least 100 trades.
2. Mean monthly net return > 0 with t >= 2.0 (monthly returns compounded from
   daily; t = mean / sd x sqrt(months)).
3. Mean monthly net return > 0 in both 2013-2019 and 2020-2026.
4. Mean net return per trade > 0 after dropping the best 3% of trades.
5. No single pair (the same two tickers across formations) above 25% of the
   absolute net P&L.

## Reported after the verdict (cannot change it)

- All 960 settings in the holdout: Spearman correlation between search and
  holdout Sharpe, and the average holdout Sharpe by search-Sharpe decile.
- The raw best setting's holdout result.
- Walk-forward: every January from 2013, re-pick by the same rule using all
  periods from 1999 to the previous December, trade that year.
- The selected setting on random cross-sector pairs in the holdout (200
  repetitions).

## Disclosures

- An earlier test (`results/pairs/`, 2026-10-23) ran one setting (60-session
  lookback, entry 2, exit at 0, stop at 4, 20-session hold) on 20 named twins and
  on newly correlated pairs over 2013-2026; both lost after costs. So 2013-2026 is
  not untouched by me, but the setting tested here is picked mechanically from
  1999-2012 and the pairs are chosen by data, point in time.
- Published work (Gatev, Goetzmann & Rouwenhorst; Do & Faff) found pairs
  profits falling after 2002 and strongest in the 2000-02 and 2007-09 bear
  markets; the search period contains both, the holdout neither. The quant
  program's H2 (Johansen pairs, 2006-2024) failed.
- Industry labels are Sharadar's last known ones (a mild lookahead: a 2026 label
  is used for a 1999 formation).
- The flat 0.15% cost is optimistic before decimalization (2001) and probably
  generous for large caps after 2013.
- An earlier message to Luis said the grid was "about 480" settings; with both
  pair methods it is 960.
