# Out-of-sample test of the tight-spread lead — pre-registration

Written and committed on 2026-10-22, before any option data for the test
periods was downloaded. Nothing below may change after results are seen. The
only number allowed to change before the real run is `TOP_N`, and only to cut
cost after the free price quote (which returns no prices or outcomes); if it
changes, the change and the quote are recorded at the bottom of this file
before the run.

Code: `replay/oos_spread.py` (sample, measurement and scoring), contract
picking from `replay/calm_test.py` and the V3 sell rules from
`replay/exit_paths.v3_rule`, both unchanged from the 2025 runs.
Workflow: `11 · Out-of-sample spread test`.

## Why

The 2025 spread check (`replay/spread_check.py`) found that options bought
with an entry bid-ask spread under 5% averaged +10.8% per trade under the live
V3 sell rules (Jan-Jun +7.4%, Jul-Dec +14.8%; 86 options), while every 5-20%
band lost. The 5% line was drawn after looking at that same 2025 data, so it
has not been tested. This runs the identical rule on periods it has never seen.

## The rule (unchanged from 2025)

- Entry: the first trading day after each weekly as-of date (the last trading
  day of each week), at 9:46 ET, on the last live 1-minute quote at or before
  9:46 (quotes from 9:31 on).
- For each sampled stock, both the call and the put the agent's contract rule
  picks: first expiry 14-21 calendar days out, else the first 12-24 days out;
  strike nearest 1% out of the money among strikes 0-3% out of the money
  (widened to 0-5% if none), standard contracts only.
- Qualifies if `(ask - bid) / mid < 5%` and Black-Scholes delta (implied vol
  from the 9:46 mid) is at least 0.35.
- Bought at the ask. Live V3 sell rules every minute, triggering on the mid
  and selling at the bid: stop -75%, ceiling +200%, trailing stop 20 points
  below the peak once the peak reaches +50%, force exit from 13:00 ET on the
  day before expiry.
- Return per trade = (exit bid - ask) / ask.

## Sample

Each week, from S&P 500 members as of the as-of date (Sharadar, point in
time):
- the 40 members with the highest average daily dollar volume (close x volume)
  over the 20 trading days ending on the as-of date (`TOP_N = 40`), and
- 10 other members drawn at random, seeded by the as-of date (as in 2025).

The weekly list is written to `results/oos/universe.csv` before any option data
is pulled.

## Periods

| Period | As-of dates | Entries | Weeks | Second half starts |
|---|---|---|---|---|
| 2024 | 2024-01-05 .. 2024-12-27 | 2024-01-08 .. 2024-12-30 | 52 | as-of 2024-07-01 |
| 2026 | 2026-01-02 .. 2026-09-25 | 2026-01-05 .. 2026-09-28 | 39 | as-of 2026-05-15 |

2026 stops at the last week whose options have finished trading by
2026-10-22.

## Primary measure

The average V3 return per qualifying trade in each period, all sampled
stocks and both sides pooled (a call and a put on the same stock-week are two
trades).

## Pass criteria (each period separately)

1. The average is above zero, and the low end of its 90% interval is above
   zero (bootstrap resampling whole weeks, 20,000 draws, seed 0).
2. The average is above zero in both halves of the period.
3. The average is still above zero after dropping the best 3% of trades.
4. The average is still above zero after dropping the single best calendar
   month (the entry month with the largest total return).

A period with fewer than 100 qualifying trades gets no verdict
("underpowered").

## Verdict

- Both periods pass all four: **confirmed**.
- One period passes, the other fails: **not confirmed**.
- Neither passes: **rejected**.
- Either period underpowered (and the other not a pass): **underpowered**.

## Reported but not part of the verdict

The same qualifying trades under V3 + day-2 sell, day-2 sell only, hold to
the Thursday exit and a +50% take-profit; the 5-20% spread options in the same
sample under V3; liquid vs random group; calls vs puts; one side per
stock-week following SPY's 20-day trend; spread cut-offs of 3/4/6/8/10%; the
most-traded tickers. None of these can turn a failed test into a pass.

## Changes before the run

None to the rule, sample or criteria. `TOP_N` stays at 40.

Free cost quote (workflow 11 run 39853163498, quote only, no option prices
downloaded): bars + option definitions for all 91 weeks $18.91; option quotes
estimated at about $4 more (from 2025's cost per quote-minute). Already spent
$8.53 of the $40 cap. Luis chose to run the test as locked on 2026-10-22.
