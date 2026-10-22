"""
Smoke test for replay/oos_spread.py on synthetic data. No API keys, no network.

  * simulate_1m with the V3 settings must match exit_paths.v3_rule exactly on
    random price paths (so the secondary rules share the primary's mechanics);
  * universe -> one shard per period -> merge -> score runs end to end, with
    ticker-dependent spreads so both tight and wide options exist.

    python tests/oos_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="oos_smoke_"))
os.environ.update({"REPLAY_DATA_DIR": str(TMP / "data"), "REPLAY_RESULTS_DIR": str(TMP / "results"),
                   "REPLAY_DATA_START": "2023-06-01", "REPLAY_DATA_END": "2026-10-23",
                   "REPLAY_CAL_START": "2023-06-01", "REPLAY_CAL_END": "2026-12-31"})
sys.path.insert(0, str(ROOT))

from replay import asof, config, dbn  # noqa: E402

rng = np.random.default_rng(11)
TICKERS = [f"T{chr(65 + i)}" for i in range(14)] + ["BRK.B"]     # roots without digits (pick() skips those)


def build_sharadar():
    days = pd.bdate_range(config.DATA_START, config.DATA_END)
    rows = []
    for k, t in enumerate(TICKERS + ["SPY"]):
        s0 = 40 + 30 * k
        close = s0 * np.exp(np.cumsum(rng.normal(0.0003, 0.017, len(days))))
        vol = 1e6 * (1 + k) * (1 + rng.random(len(days)))
        for i, d in enumerate(days):
            rows.append((t, d, close[i], close[i] * 1.01, close[i] * 0.99, close[i], vol[i], close[i], close[i]))
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close", "volume",
                                     "closeadj", "closeunadj"])
    config.SHARADAR_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SHARADAR_DIR / "prices.parquet", index=False)
    snaps = [(q, "historical", t) for q in pd.date_range("2023-03-31", "2026-09-30", freq="QE") for t in TICKERS]
    pd.DataFrame(snaps, columns=["date", "action", "ticker"]).to_parquet(config.SHARADAR_DIR / "sp500.parquet",
                                                                         index=False)


_bars = {}


def fake_equity_bars(ticker, d, purpose="x"):
    key = (ticker, d)
    if key not in _bars:
        raw = asof.prices()[ticker.replace("-", ".")]
        if pd.Timestamp(d) not in raw.index:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        c = float(raw.loc[pd.Timestamp(d), "closeunadj"])
        idx = pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min")
        path = c * np.exp(np.cumsum(rng.normal(0, 0.0012, len(idx))))
        _bars[key] = pd.DataFrame({"open": path, "high": path, "low": path, "close": path, "volume": 100.0},
                                  index=idx)
    return _bars[key]


def fake_defs_for_day(d, tickers, purpose=""):
    out = {}
    for u in tickers:
        S = asof.unadjusted_close(u, asof.prev_trading_day(d))
        exps = [d + timedelta(days=k) for k in range(1, 40) if (d + timedelta(days=k)).weekday() == 4]
        rows = []
        for e in exps:
            for K in np.arange(round(S * 0.9), round(S * 1.1) + 1, 1.0):
                for c in "CP":
                    root = u.replace("-", "").replace(".", "")
                    rows.append({"raw_symbol": f"{root:<6}{e:%y%m%d}{c}{int(K * 1000):08d}", "expiration": e,
                                 "instrument_class": c, "strike_price": K})
        out[u] = pd.DataFrame(rows)
    return out


def fake_quotes(symbols, start, end, purpose):
    from replay.agent import bs_price
    out = []
    for sym in symbols:
        root, e, call, K = sym[:6].strip(), pd.Timestamp("20" + sym[6:12]).date(), sym[12] == "C", int(sym[13:]) / 1000
        u = "BRK-B" if root == "BRKB" else root
        half = 0.005 + 0.04 * ((ord(root[1]) - 65) % 5 / 4 if root[0] == "T" else 0.5)   # 1%..9% wide
        for d in asof.trading_days(pd.Timestamp(start).date(), min(pd.Timestamp(end).date(), e)):
            b = fake_equity_bars(u, d)
            b = b[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
            for ts, S in b["close"].items():
                T = max((pd.Timestamp(e) + pd.Timedelta(hours=16) - ts).total_seconds() / 3.15e7, 1e-4)
                m = bs_price(S, K, T, 0.045, 0.35, call)
                out.append({"symbol": sym, "ts_et": ts, "bid": m * (1 - half), "ask": m * (1 + half)})
    return pd.DataFrame(out)


dbn.equity_bars = fake_equity_bars
dbn.prefetch_equity_day = lambda *a, **k: None
dbn.option_quotes = fake_quotes
dbn.ledger = lambda: {"spent_usd": 0.0, "quoted_usd": 0.0}

from replay import exit_paths, oos_spread as o  # noqa: E402

o.defs_for_day = fake_defs_for_day
o.TOP_N, o.RANDOM_N, o.MIN_TRADES = 6, 3, 5
o.PERIODS = {"2024": ("2024-01-05", "2024-01-26", "2024-01-15"),
             "2026": ("2026-01-02", "2026-01-23", "2026-01-12")}


def check_v3_equivalence(n=400):
    days = asof.trading_days("2024-03-04", "2024-03-15")
    expiry = date(2024, 3, 15)
    for k in range(n):
        idx = [pd.Timestamp(f"{d} 09:31") + pd.Timedelta(minutes=m) for d in days[:-1] for m in range(0, 390, 7)]
        mid = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.03 + 0.03 * (k % 3), len(idx))))
        p = pd.DataFrame({"ts_et": idx, "mid": mid, "bid": mid * 0.98})
        cost = 2.0 * 1.02
        a = exit_paths.v3_rule(p, cost, expiry)["v3_pct"]
        b = o.simulate_1m(p, cost, expiry, days[0], **o.V3)
        assert abs(a - b) < 1e-9, (k, a, b)
    print(f"simulate_1m == v3_rule on {n} random paths")


def main():
    build_sharadar()
    check_v3_equivalence()
    u = o.build_universe()
    assert set(u["group"]) == {"liquid", "random"} and u.groupby("asof").size().eq(o.TOP_N + o.RANDOM_N).all()
    assert u[u["group"] == "liquid"].groupby("asof")["dv_rank"].max().eq(o.TOP_N).all()
    o.shard(0, 2)
    o.shard(1, 2)
    dbn._LEDGER_REPO.parent.mkdir(parents=True, exist_ok=True)
    dbn._LEDGER_REPO.write_text('{"spent_usd": 0.0}')
    o.merge()
    t = pd.read_csv(o.TRADES)
    q = t[t["qualifies"] == True]  # noqa: E712
    w = t[(t["tradable"] == True) & (t["spread_pct"] >= 5)]  # noqa: E712
    print(f"rows {len(t)} | qualifying {len(q)} | wide tradable {len(w)} | skips {t['skip'].value_counts().to_dict() if 'skip' in t else {}}")
    assert len(q) and len(w) and q["v3_pct"].notna().all() and (q["spread_pct"] < 5).all()
    assert (q["delta"] >= 0.35).all()
    out = o.score()
    print({p: (v["verdict"], v.get("n"), v.get("mean")) for p, v in out["periods"].items()}, out["verdict"])
    print(f"\nOOS SMOKE OK - outputs in {TMP}")


if __name__ == "__main__":
    main()
