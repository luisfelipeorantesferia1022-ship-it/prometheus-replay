# Twin-stock pairs test — pre-registration

Written and committed on 2026-10-23, before any trade in the test period was
simulated. Only a mechanics check on 2010-07 to 2012-12 was run (trade counts,
exit reasons, holding times, pairs picked at each formation); it printed no
returns. Nothing below may change after results are seen.

Code: `replay/pairs_test.py`. Data: Sharadar daily prices (adjusted closes, opens
on the same basis), point-in-time S&P 500 membership, from Luis's bulk files
(`quant/data/sharadar/derived/`, not committed). Stocks only: no options, V3 does
not apply.

## Why

Luis asked whether "twin" stocks that usually move together (Coca-Cola and
Pepsi, Home Depot and Lowe's, Visa and Mastercard, ...) can be traded when one
falls behind: buy the laggard, short the leader, wait for them to come back
together. And whether new pairs are forming that could be traded the same way.

## Variants (2 trials)

**P1 twins:** 20 pairs chosen in advance on economic grounds (same industry, same
customers): Luis's six, KO/PEP, HD/LOW, V/MA, WMT/TGT, NKE/UAA, UPS/FDX, and
XOM/CVX, JPM/BAC, GS/MS, MCD/YUM, LMT/NOC, CSX/NSC, DUK/SO, PG/CL, MO/PM, T/VZ,
ADP/PAYX, SPGI/MCO, HLT/MAR, DAL/UAL. A pair only trades while both stocks are in
the S&P 500.

**P2 emerging:** on the first session of every January and July, among the 150
S&P 500 members with the most dollar volume over the prior 60 sessions, pick
pairs whose daily returns were tightly linked over the past year (correlation
>= 0.70 over 252 sessions) and became so recently (at least 0.20 higher than over
the 756 sessions before that, or too new to have 500 sessions of that history).
Share classes of one company are excluded. The 20 most correlated are traded for
the next six months; open positions close at the next formation.

## Trading rule (both variants)

- Spread = ln(price A) - ln(price B); z = (spread - its 60-session mean) / its
  60-session standard deviation, at each close.
- Entry: when |z| >= 2 at a close, at the next open buy the cheap stock and short
  the rich one, equal dollars.
- Exit at the next open after any of: z crosses 0 (converged); |z| reaches 4
  (stop); 20 sessions held; either stock leaves the S&P 500; the P2 period ends.
- After any exit a pair needs |z| < 1 before it can open again.
- Return per trade on gross notional = (long leg - short leg) / 2, minus 0.15%
  round-trip cost (both legs, the quant program's cost model) and 0.5% a year
  borrow on the short half.
- Test period: entries from 2013-01-02 to the end of the data (2026-10-08).
  Halves: entries before 2020-01-01 and from 2020-01-01.

## Pass bar (each variant; all five must hold)

1. At least 100 trades.
2. Average net return per trade > 0 and t >= 2.0, clustered by entry week.
3. Average net return > 0 in both halves.
4. Still > 0 after dropping the best 3% of trades.
5. No single pair accounts for more than 25% of the absolute net P&L.

## Reported but not part of the verdict

Gross (before cost) results, win rate, exit reasons, holding time, results by
year and, for P1, by pair (with Luis's six marked), and the pairs P2 picked at
each formation.

## Disclosures

Seen before writing this: daily-return correlations by period and the change in
price ratio since 2010 for KO/PEP, HD/LOW, V/MA, WMT/TGT, NKE/UAA, UPS/FDX,
XOM/CVX, JPM/BAC and GOOGL/GOOG; a screen of the tightest and most newly tight
liquid pairs over Oct 2024-Oct 2026, in which GS/MS appeared; the quant program's
H2 (within-sector Johansen pairs) failing, net -0.46% per trade on Sharadar;
published results showing pairs profits declining after 2002. No return of any
trade under this rule in the test period has been computed.
