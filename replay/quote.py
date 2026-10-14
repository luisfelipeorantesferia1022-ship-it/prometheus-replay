"""
Price every Databento purchase the 2025 replay needs, without buying anything.

Only metadata endpoints are called (get_cost, get_dataset_range,
symbology.resolve). Databento does not bill metadata calls, so running this
spends nothing. Standalone: needs only `pip install databento pandas`.

    set DATABENTO_API_KEY=...        (PowerShell: $env:DATABENTO_API_KEY="...")
    python quote.py

Three things are priced from small sample requests, then scaled:

  1. Stage 2: 1-minute stock bars, 09:00-16:00 ET, for every weekly candidate
     over the 60 days before each weekend. Unit = one ticker-day.
  2. Agent entries: the option chain definitions for an underlying on an entry
     morning, plus 1-minute quotes for the ~20 contracts near the money.
  3. Agent holds: 1-minute quotes and trades for each contract held, from entry
     to its expiry (worst case; most exit days earlier).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

try:
    import databento as db
except ImportError:
    sys.exit("pip install databento pandas")

KEY = os.environ.get("DATABENTO_API_KEY", "").strip()
if not KEY:
    sys.exit("set DATABENTO_API_KEY first")
C = db.Historical(KEY)
ET = "America/New_York"

SAMPLE_DAY = "2025-03-12"            # an ordinary Wednesday
SAMPLE_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "UNH", "EOG", "LEN",
                  "NVR", "TFC", "PEG", "ATO", "COST", "MDT", "DLR", "CPRT",
                  "GPN", "KHC", "CARR", "ETN", "AFL", "BA", "TSCO", "DLTR", "UBER"]
OPTION_SAMPLES = ["AAPL", "NVDA", "UNH", "EOG", "LEN", "ATO"]

# Volume scenarios (counts, not dollars)
STAGE2_TICKER_DAYS = {"low": 12_000, "mid": 25_000, "high": 55_000}
#   high = 52 weeks x 25 candidates x 42 trading days with no overlap at all
ENTRY_DAYS_WITH_ATTEMPTS = {"low": 120, "mid": 250, "high": 450}
#   distinct (underlying, morning) chain lookups across copy + real + control
HELD_CONTRACTS = {"low": 150, "mid": 300, "high": 500}


def t(day, hm):
    return pd.Timestamp(f"{day} {hm}", tz=ET).tz_convert("UTC")


def cost(**kw) -> float:
    return float(C.metadata.get_cost(**kw))


def dataset_ok(ds: str) -> bool:
    try:
        r = C.metadata.get_dataset_range(dataset=ds)
        start = r["start"] if isinstance(r, dict) else r.start
        print(f"  {ds}: history from {str(start)[:10]}")
        return pd.Timestamp(start).tz_localize(None) <= pd.Timestamp("2024-09-01")
    except Exception as e:
        print(f"  {ds}: unavailable ({type(e).__name__}: {e})")
        return False


def main():
    print("Equity datasets:")
    eq = next((d for d in ["EQUS.MINI", "XNAS.ITCH"] if dataset_ok(d)), None)
    if not eq:
        sys.exit("no equity dataset covers late 2024")
    print(f"  using {eq}\n")

    # ── 1. Stage 2 stock bars ───────────────────────────────────────────────
    c_day = cost(dataset=eq, schema="ohlcv-1m", symbols=SAMPLE_TICKERS, stype_in="raw_symbol",
                 start=t(SAMPLE_DAY, "09:00"), end=t(SAMPLE_DAY, "16:01"))
    c_month = cost(dataset=eq, schema="ohlcv-1m", symbols=SAMPLE_TICKERS, stype_in="raw_symbol",
                   start=t("2025-03-03", "09:00"), end=t("2025-03-31", "16:01"))
    per_td = max(c_day / len(SAMPLE_TICKERS), c_month / (len(SAMPLE_TICKERS) * 21))
    print(f"1. Stage 2 bars: ${c_day:.4f} for 25 tickers x 1 day, ${c_month:.4f} for 25 x 21 days")
    print(f"   -> ${per_td:.5f} per ticker-day")

    # ── 2 & 3. Options ──────────────────────────────────────────────────────
    defs, chain, held = [], [], []
    for u in OPTION_SAMPLES:
        try:
            cd = cost(dataset="OPRA.PILLAR", schema="definition", symbols=[f"{u}.OPT"],
                      stype_in="parent", start=t(SAMPLE_DAY, "00:00") , end=t(SAMPLE_DAY, "09:30"))
            res = C.symbology.resolve(dataset="OPRA.PILLAR", symbols=[f"{u}.OPT"],
                                      stype_in="parent", stype_out="instrument_id",
                                      start_date=SAMPLE_DAY, end_date="2025-03-13")
            r = (res.get("result") or {}) if isinstance(res, dict) else {}
            n = max(1, len(r[f"{u}.OPT"]) if f"{u}.OPT" in r else len(r))
            cq = cost(dataset="OPRA.PILLAR", schema="cbbo-1m", symbols=[f"{u}.OPT"], stype_in="parent",
                      start=t(SAMPLE_DAY, "09:30"), end=t(SAMPLE_DAY, "16:00"))
            ct = cost(dataset="OPRA.PILLAR", schema="trades", symbols=[f"{u}.OPT"], stype_in="parent",
                      start=t(SAMPLE_DAY, "09:30"), end=t(SAMPLE_DAY, "16:00"))
        except Exception as e:
            print(f"   {u}: quote failed ({type(e).__name__}: {e})")
            continue
        # Near-the-money contracts quote and trade far more often than the
        # chain average, so the per-contract figure is scaled up 3x.
        q_contract_day = 3 * cq / n
        t_contract_day = 3 * ct / n
        defs.append(cd)
        # chain lookup: ~20 contracts, last 30 min of prior day + first ~45 min
        chain.append(20 * q_contract_day * (75 / 390))
        # held contract: entry to expiry, ~14 trading days of quotes + trades
        held.append(14 * (q_contract_day + t_contract_day))
        print(f"2. {u}: {n} contracts listed | definitions ${cd:.4f} | full-day quotes for "
              f"the whole chain ${cq:.4f} | trades ${ct:.4f}")

    if not defs:
        sys.exit("option quotes failed")
    avg_def, avg_chain, avg_held = (sum(x) / len(x) for x in (defs, chain, held))
    print(f"\n   per entry-morning lookup: definitions ${avg_def:.4f} + chain quotes ${avg_chain:.4f}")
    print(f"   per held contract (to expiry): ${avg_held:.4f}")

    # ── Totals ──────────────────────────────────────────────────────────────
    print("\nEstimated total (USD)")
    print(f"{'scenario':<8}{'stage 2 bars':>14}{'chain lookups':>15}{'held options':>14}{'agent stock bars':>18}{'TOTAL':>10}")
    for s in ("low", "mid", "high"):
        b = STAGE2_TICKER_DAYS[s] * per_td
        e = ENTRY_DAYS_WITH_ATTEMPTS[s] * (avg_def + avg_chain)
        h = HELD_CONTRACTS[s] * avg_held
        a = 252 * 8 * per_td      # ~8 tickers/day of agent stock bars, mostly cached
        print(f"{s:<8}{b:>14.2f}{e:>15.2f}{h:>14.2f}{a:>18.2f}{b + e + h + a:>10.2f}")
    print("\nMetadata calls only; nothing was downloaded or billed.")


if __name__ == "__main__":
    main()
