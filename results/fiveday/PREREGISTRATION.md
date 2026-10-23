# Five-day direction test — pre-registration

Written and committed on 2026-10-23, before any candidate was scored on the
test years. Only a mechanics check on as-of dates 2010-2012 (counts, missing
values, one stock-week verified by hand) was run before this commit; it reports
no accuracy. Nothing below may change after results are seen.

Code: `replay/fiveday_test.py`. Data: Sharadar daily prices (split- and
dividend-adjusted `closeadj`, opens put on the same basis), point-in-time S&P 500
membership, and SPY, derived from Luis's Sharadar bulk files into
`quant/data/sharadar/derived/` (not committed: licensed data).

## Why

The oracle test (`results/analysis/oracle_test.txt`) showed V3 is paid for the
stock's direction over about the next five trading days. Knowing it perfectly
picks the winning option side 84% of the time. With random errors, about 55%
accuracy breaks even, 60% makes about +4% per trade and 65% about +9%. The agent
needs about three trades a week, so what matters is the accuracy of a
forecaster's most confident calls.

## Setup

- Weeks: as-of = last trading day of each ISO week; entry = the next session's
  open (day 0); exit = the close five sessions later (day 5). Return =
  close(day 5) / open(day 0) - 1 on the adjusted basis. A stock that delists
  mid-week exits at its last close.
- Universe each week: S&P 500 members on the as-of date, price >= $5, 20-day
  average dollar volume >= $20M, at least 260 sessions of history.
- Test period: as-of dates 2013-01-04 to 2026-09-25 (the last week whose day-5
  close is in the data). Each calendar year is scored separately.
- Calls: each candidate scores every eligible stock; its 10 largest absolute
  scores that week are its calls; direction = sign of the score. A call is right
  if the stock moved that way (a return of exactly zero counts as wrong).

## Candidates (5 trials)

| | Score | Idea |
|---|---|---|
| K1 trend20 | 20-day return | follow the last month |
| K2 trend60 | 60-day return | follow the last quarter |
| K3 reversal5 | minus the 5-day return | fade last week |
| K4 regime_beta | +beta if SPY is at or above its 20-day average, else -beta | the agent's SPY rule, on the stocks that move most with the market |
| K5 gbm | P(up) - 0.5 from a gradient-boosted classifier | all 23 features combined |

K5: scikit-learn `HistGradientBoostingClassifier(max_iter=200,
learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
l2_regularization=1.0, random_state=0)`, refit each January on every stock-week
from 2010 with an as-of date at least 14 days before the test year begins.
Features: returns over 1/5/20/60/120/250 days; distance from the 20/50/200-day
averages; 20- and 60-day volatility and their ratio; 5-day vs 60-day dollar
volume; RSI(14); distance from the 52-week high; 1-year beta to SPY; SPY's 5- and
20-day return, distance from its 20-day average and 20-day volatility; and the
stock's within-week rank of its 5-day return, 20-day return and volatility.

## Pass bar (each candidate; all four must hold)

1. Pooled accuracy of its calls >= 60%.
2. Accuracy >= 55% in every calendar year 2013-2026.
3. The low end of the 90% interval of pooled accuracy (bootstrap resampling
   whole weeks, 20,000 draws, seed 0) >= 57%.
4. Accuracy exceeds that week's base rate by >= 5 points pooled, where the base
   rate is the share of all eligible stocks that moved in the call's direction
   that week. This removes market drift: calling "up" in a week when 60% of
   stocks rose is worth 60% by luck.

## What follows

- A candidate that passes is checked on the real option trades (2024-2026
  stock-weeks with both sides measured): the V3 result of the side it picks on
  the stock-weeks it calls. Reported, then a fresh pre-registered test before
  any live use.
- If none passes: no simple or combined price-based signal calls next-week
  direction well enough for V3, and the directional-options approach stops here
  unless a new kind of information (option-implied, news) is tested.

## Disclosures

Seen before writing this: the oracle test (5-day up-rate 51-54% in the
2024-2026 option names); the quant program's weekly reversal (H4: +0.095% gross
per 5-day trade, t 2.09, on Sharadar 2006-2024), which bears on K3; the 2025
LSTM accuracy having no persistence; trend-following side rules in the 2025
winner search (positive in both halves, weak in the second). No accuracy of any
candidate on these test years has been computed.
