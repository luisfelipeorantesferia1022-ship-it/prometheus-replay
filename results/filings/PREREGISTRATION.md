# Do changes in 10-K/10-Q filings, read by an LLM, predict monthly returns? Pre-registration

Written and committed on 2026-10-23, before any filing was downloaded and before
any return linked to any filing signal was computed. Nothing in the sections
"Signals", "Portfolios", "Periods", "Selection", "Luck checks", "Holdout and pass
bar" or "Leakage checks" may change after this commit. Data-engineering fixes
(downloading, HTML parsing, section finding, masking) are allowed until the first
return is joined to any signal, and each one is listed under "Amendments" with
its commit and the reason, together with what had been looked at (parse
statistics only, never returns).

Decisions made by Luis on 2026-10-23: model Claude Haiku 5.5 with a hard $250
cap; search 2006-2018, holdout 2019-2026; S&P 500 point in time; primary
portfolio long-short, industry-neutral.

## The question

Cohen, Malloy & Nguyen ("Lazy Prices", JF 2020, data 1995-2014) found that firms
whose 10-K/10-Q text changes a lot from the year before underperform, with no
announcement reaction and returns that keep accruing for months. Published
anomalies lose about half their return after publication (McLean & Pontiff). The
bet here: an LLM judging what the changes mean (worse risk, new litigation,
resolved problems) predicts returns better than word-count similarity, and well
enough to trade monthly in the S&P 500 after costs.

## Power, stated in advance

At the decayed Lazy Prices size (about 0.15-0.25% a month for a long-short at
about 2.5% monthly volatility), a 93-month holdout gives an expected t-stat near
1. This test can only pass if the LLM signal is materially stronger than the
published word-similarity effect. A failure therefore says "no strong edge here",
not "the Lazy Prices effect is zero".

## Data

- Universe of firms: every Sharadar ticker that was an S&P 500 member at any time
  from 2005-01-01 to 2026-10-08 (963 tickers). SEC CIK from Sharadar's
  `tickers.secfilings` link (956 tickers; the 7 without a CIK are second share
  classes: CMCSK, DISCK, FOX, GOOG, NWS, TFCFA, UA, dropped so a company is not
  counted twice). List: `results/filings/universe_ciks.csv`.
- Filings: EDGAR `data.sec.gov/submissions` for each CIK; forms 10-K and 10-Q
  only (no amendments, no 10-KT, no 10-K405 since none fall in range); filed
  2004-01-01 to 2026-10-23; the primary document only (exhibits excluded).
  Downloaded on GitHub Actions with a declared User-Agent, at most 10 requests a
  second in total.
- Filing to ticker: a filing belongs to the ticker with that CIK whose Sharadar
  price life (first to last price date) contains the acceptance date. If two
  tickers with the same CIK both qualify (MRK/SGP1 before 2009-11, JCI/TYC,
  CHKAQ/EXE, and similar), the filing is dropped.
- Timing: EDGAR `acceptanceDateTime`, converted to the New York calendar date.
- Prices, membership, sectors and fundamentals: Luis's Sharadar bulk files on his
  PC (licensed, never committed). Monthly total returns from `closeadj`.

## Text processing (reference implementation `replay/filings_text.py`)

- HTML to text with line breaks at block elements; hidden inline-XBRL header
  removed; non-breaking spaces normalized.
- Sections (heading regex at line start; a heading followed by another heading of
  the same item before the end heading gives way to the later one; of the rest,
  the longest span wins, which skips the table of contents; a section under 500
  characters counts as absent):
  - 10-K: Item 1A Risk Factors (to Item 1B or 2), Item 3 Legal Proceedings (to
    Item 4), Item 7 MD&A (to Item 7A or 8).
  - 10-Q: Part II Item 1A Risk Factors (to Item 2), Part II Item 1 Legal
    Proceedings (to Item 1A or 2), Part I Item 2 MD&A (to Item 3 or 4).
- Comparison filing ("prior"): the same form of the same CIK whose period of
  report is 330 to 400 days before this filing's period of report (same fiscal
  quarter a year earlier). No prior, no signal.
- Words: lowercase `[a-z]+` tokens, nothing removed.
- Paragraphs: text blocks between line breaks with at least 8 words. Matching
  uses each paragraph's word set after lowercasing and replacing digits with `#`.
  Each current paragraph is matched to its most similar prior paragraph by
  Jaccard: >= 0.9 unchanged, 0.5 to 0.9 modified (paired with that prior
  paragraph), < 0.5 added. Prior paragraphs with no current paragraph at >= 0.5
  are removed.

## Signals

All four are oriented so that higher means "expected to do better".

- **F1 full-document similarity:** cosine similarity of word-count vectors of the
  whole primary document vs. the prior. (Lazy Prices replication; no LLM.)
- **F2 risk-factor similarity:** the same on the Risk Factors section; missing if
  either filing lacks the section.
- **F3 LLM change score:** Claude Haiku 5.5 (`claude-haiku-5-5`, temperature 0,
  Batch API) reads the added, removed and modified paragraphs of Risk Factors,
  Legal Proceedings and MD&A (numbers kept; caps of 12,000 characters for Risk
  Factors, 6,000 for Legal, 12,000 for MD&A; within a section added paragraphs
  first, then modified, then removed, each in document order, cut at a paragraph
  boundary with a note of how much was left out). Masking before the model sees
  anything: company names (EDGAR current and former names, with and without
  suffixes such as Inc, Corp, Co, plc, Holdings), the company's tickers, years
  1950-2039 -> `[YEAR]`, full dates -> `[DATE]`, `Mr./Ms./Mrs./Dr. Name` ->
  `[PERSON]`. The system prompt tells the model to judge only the text and not
  to use outside knowledge. Output JSON with `tone` (integer -3 to +3: -3 major
  adverse new information about risks or prospects, 0 neutral or cosmetic, +3
  major favourable), `materiality` (0-3), seven yes/no flags (new material risk,
  litigation escalated, litigation resolved, liquidity or going concern,
  management or auditor change, restructuring or impairment, government
  investigation) and a short summary. **F3 = tone.** A filing where at least one
  of the three sections is found in both reports but no section has a changed
  paragraph gets tone 0 without a call; if no section is found in both reports,
  F3 is missing. A reply that
  does not parse is retried once; if it fails again, F3 is missing. Prompt text:
  `replay/filings_llm.py`, frozen with this commit.
- **F4 gated LLM score:** F3 where the filing's F1 percentile (below) is at most
  0.4 (a real wording change); otherwise the neutral value 0.5.

Point-in-time scaling (all four): each filing's raw value becomes its percentile
rank (ties at the midrank) among filings of the same form type accepted in the
365 days up to and including its own acceptance date (at least 50 filings; if
fewer, all earlier filings of that form). This removes 10-K vs 10-Q level
differences and the long-run growth in filing length. F4 uses F3's percentile.

## Portfolios

- Formation date t: the last NYSE session of each month. Eligible stock: S&P 500
  member on t, a price on t, unadjusted close >= $5, and a filing with the signal
  accepted on a calendar date before t and no more than 91 days before t (the
  most recent such filing is used).
- Sectors: Sharadar `sector` (11 sectors; current label, the only one
  available). A sector with fewer than 10 eligible stocks that month is merged
  into one "other" group.
- Within each sector group, percentile-rank the signal; long the top 20%, short
  the bottom 20%, equal weight, each leg scaled to $1.
- Holding: **K = 1** (rebalance monthly) or **K = 3** (Jegadeesh-Titman: the
  month's weights are the average of the last three formations' weights).
- Return in month t+1: from the close on t to the close on the last session of
  month t+1, Sharadar adjusted closes. A stock that stops trading earns its
  return to its last close and then cash.
- Costs: 0.15% per round trip, charged as 0.075% x sum of absolute weight changes
  at each formation (both legs), plus 0.5% a year borrow on the short leg. The
  same costs as the pairs tests.

Eight settings: {F1, F2, F3, F4} x {K = 1, K = 3}.

## Periods

- **Search:** return months 2007-02 to 2018-12 (143 months). The start allows
  every signal a prior-year comparison: Item 1A Risk Factors became mandatory for
  fiscal years ending after 2005-12-01.
- **Holdout:** return months 2019-01 to 2026-09 (93 months). Halves: 2019-01 to
  2022-11 and 2022-12 to 2026-09. Looked at once, after the selection is
  committed.
- The LLM scores every filing in both periods in one pass before the selection
  (it has to, to score the search). Holdout signals are computed but not joined
  to holdout returns until the selection commit exists.

## Selection (search period only)

The setting with the highest annualized Sharpe ratio of net monthly long-short
returns. Ties: lower average turnover. The selection, the search table and the
luck checks are committed before any holdout return is computed.

## Luck checks (search period, reported with the selection)

- White's Reality Check over the 8 settings (stationary bootstrap, mean block 6
  months, 5,000 draws) on net monthly returns.
- Deflated Sharpe ratio of the selected setting with N = 8 and the variance of
  the 8 Sharpes. Context: this research history has many more trials (see
  `claude/research-log.md`), but they asked different questions.
- Placebo: 500 runs in which each month's signal values are shuffled across
  stocks within each sector; the share of runs whose best-of-8 Sharpe is at least
  the selected setting's.
- Gross (zero-cost) results and a value-weighted version are reported, not used
  for selection.

## Holdout and pass bar (selected setting, one look)

**Signal verdict:** Fama-MacBeth regressions each holdout month of the stock's
next-month return on its signal percentile (within-sector, centered at 0) with
sector dummies and percentile-ranked controls: log market cap (unadjusted close x
Sharadar `sharesbas` from the latest quarterly row filed before t), log
book-to-market (latest quarterly `equity` filed before t over market cap),
momentum (return t-12 to t-1 months) and last month's return (missing controls
set to the month's median). Pass: mean slope t-stat >= 2.0 (Newey-West, 3 lags)
and a positive mean slope in both halves.

**Strategy verdict:** net long-short monthly returns. Pass requires all of:
1. Mean t-stat >= 2.0 (Newey-West, 3 lags).
2. Positive mean in both halves.
3. Positive mean after removing the best 3% of months.
4. No single stock contributes more than 25% of total net profit.

Outcome: **PASS** if the strategy verdict passes; **SIGNAL ONLY** if only the
signal verdict passes (real but not tradable as built); **FAIL** otherwise.
FAIL ends the project (kill rule). All 8 settings' holdout results are reported
as well, labelled exploratory.

## Leakage checks

Haiku 5.5's training data runs to June 2026, so it has seen news about every
period tested. Masking reduces, but cannot remove, the chance that it recognises
a company and scores with hindsight.

- **L1 identification:** 300 random holdout filings, masked payload, the model is
  asked to name the company or say "unknown". Report the share named correctly.
- **L2 masked vs unmasked:** 600 random holdout filings scored a second time
  without masking. Report the rank correlation of each version's score with the
  next three months' within-sector return. A clearly higher correlation unmasked
  means hindsight leaks through names, and the masked holdout result is then an
  upper bound.
- **L3 forward test:** if the outcome is PASS or SIGNAL ONLY, score every
  S&P 500 10-K/10-Q filed from 2026-11-01 (after the model's training cutoff) the
  same way and evaluate the frozen rule after 12 months. Only this test is free
  of hindsight.

## Budget

Hard cap $250 of Anthropic API spend for this project, including the pilot and
the leakage checks, enforced in code from the token counts the API reports
(priced at list without the batch discount, so the guard is conservative).
Pilot first: 300 random search-period filings, used only to check parsing of the
replies, score distribution and cost, never returns.

## Time box

Four to six weeks from 2026-10-23. Kill if the holdout outcome is FAIL.

## Amendments

(none yet)
