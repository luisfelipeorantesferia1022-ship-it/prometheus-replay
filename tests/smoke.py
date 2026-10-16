"""
End-to-end smoke test on synthetic data. No API keys, no network.

Builds a fake Sharadar subset and a fake Databento (Black-Scholes quotes on a
random-walk tape), shortens the LSTM to one epoch, and runs every stage over
two weeks of January 2025. It checks plumbing, not results.

    python tests/smoke.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="replay_smoke_"))
os.environ["REPLAY_DATA_DIR"] = str(TMP / "data")
os.environ["REPLAY_RESULTS_DIR"] = str(TMP / "results")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategy"))

from replay import config  # noqa: E402

config.REPLAY_FIRST_ASOF = "2025-01-03"
config.REPLAY_LAST_ASOF = "2025-01-10"
config.AGENT_LAST_ENTRY_DAY = "2025-01-17"
config.AGENT_LAST_EXIT_DAY = "2025-02-14"
config.STAGE2_MIN_GAIN_PCT = 0.0   # random tape: let any positive ticker through

from replay import asof, dbn  # noqa: E402

TICKERS = ["AAA", "BBB", "CCC", "BRK.B"]
rng = np.random.default_rng(7)


# ── Fake Sharadar ────────────────────────────────────────────────────────────
def build_sharadar():
    days = pd.bdate_range(config.DATA_START, config.DATA_END)
    rows = []
    for t, s0 in zip(TICKERS + ["SPY"], [50, 120, 300, 410, 450]):
        r = rng.normal(0.0004, 0.018, len(days))
        close = s0 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.004, len(days)))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.006, len(days))))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.006, len(days))))
        div = np.where(days < pd.Timestamp("2025-06-01"), 0.985, 1.0)
        for i, d in enumerate(days):
            rows.append((t, d, open_[i], high[i], low[i], close[i], 1e6 * (1 + rng.random()),
                         close[i] * div[i], close[i]))
    df = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close",
                                     "volume", "closeadj", "closeunadj"])
    d = config.SHARADAR_DIR
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "prices.parquet", index=False)
    snaps = []
    for q in pd.date_range("2023-03-31", "2025-12-31", freq="QE"):
        for t in ["AAA", "BBB", "CCC"]:
            snaps.append((q, "historical", t))
    snaps.append((pd.Timestamp("2025-01-02"), "added", "BRK.B"))
    pd.DataFrame(snaps, columns=["date", "action", "ticker"]).to_parquet(d / "sp500.parquet", index=False)
    ev = [(t, pd.Timestamp(x)) for t in TICKERS for x in ["2024-10-25", "2025-01-29", "2025-04-28"]]
    pd.DataFrame(ev, columns=["ticker", "date"]).to_parquet(d / "earnings.parquet", index=False)


# ── Fake Databento ───────────────────────────────────────────────────────────
_bars = {}


def fake_equity_bars(ticker, d, purpose="x"):
    key = (ticker, d)
    if key not in _bars:
        raw = asof.prices()[ticker.replace("-", ".")]
        if pd.Timestamp(d) not in raw.index:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        row = raw.loc[pd.Timestamp(d)]
        idx = pd.date_range(f"{d} 09:00", f"{d} 15:59", freq="1min")
        path = np.linspace(row["open"], row["closeunadj"], len(idx)) * \
            np.exp(np.cumsum(rng.normal(0, 0.0005, len(idx))) * np.linspace(1, 0, len(idx)))
        _bars[key] = pd.DataFrame({"open": path, "high": path * 1.0005, "low": path * 0.9995,
                                   "close": path, "volume": 100.0}, index=idx)
    return _bars[key]


def fake_defs(underlying, d):
    S = asof.unadjusted_close(underlying, asof.prev_trading_day(d))
    exps = [d + timedelta(days=k) for k in range(1, 50) if (d + timedelta(days=k)).weekday() == 4]
    rows = []
    for e in exps:
        for K in np.arange(round(S * 0.9), round(S * 1.1) + 1, 1.0):
            for c in "CP":
                rows.append({"raw_symbol": f"{underlying:<6}{e:%y%m%d}{c}{int(K * 1000):08d}",
                             "expiration": e, "instrument_class": c, "strike_price": K,
                             "_u": underlying})
    return pd.DataFrame(rows)


def _parse(sym):
    u = sym[:6].strip()
    e = pd.Timestamp("20" + sym[6:12]).date()
    return u, e, sym[12] == "C", int(sym[13:]) / 1000


def fake_quotes(symbols, start, end, purpose):
    from replay.agent import bs_price
    out = []
    for sym in symbols:
        u, e, call, K = _parse(sym)
        for d in asof.trading_days(pd.Timestamp(start).date(), min(pd.Timestamp(end).date(), e)):
            b = fake_equity_bars(u, d).between_time("09:30", "15:59")
            b = b[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
            for ts, S in b["close"].iloc[::5].items():
                T = max((pd.Timestamp(e) + pd.Timedelta(hours=16) - ts).total_seconds() / 3.15e7, 1e-4)
                m = bs_price(S, K, T, 0.045, 0.30, call)
                out.append({"symbol": sym, "ts_et": ts, "bid": m * 0.97, "ask": m * 1.03})
    return pd.DataFrame(out)


def fake_trades(sym, start, end, purpose):
    q = fake_quotes([sym], start, end, purpose)
    if q.empty:
        return q
    return q.assign(price=(q["bid"] + q["ask"]) / 2)[["ts_et", "price"]]


dbn.equity_bars = fake_equity_bars
dbn.prefetch_equity_day = lambda *a, **k: None
dbn.option_definitions = fake_defs
dbn.option_quotes = fake_quotes
dbn.option_trades = fake_trades
dbn.ledger = lambda: {"spent_usd": 0.0}


def fast_models():
    import lstm_test
    import lstm_forecast
    from tensorflow.keras.models import Sequential

    def fast_train(scaled, train_len):
        x = np.array([scaled[i - 90:i] for i in range(90, train_len)])
        y = scaled[90:train_len, 0]
        m = Sequential([__import__("tensorflow").keras.layers.Input((90, scaled.shape[1])),
                        __import__("tensorflow").keras.layers.LSTM(8),
                        __import__("tensorflow").keras.layers.Dense(1)])
        m.compile("adam", "mse")
        m.fit(x, y, epochs=1, verbose=0, batch_size=256)
        return m
    lstm_test._orig_train_model = fast_train

    class FastSeq(Sequential):
        def fit(self, *a, **k):
            k["epochs"] = 1
            return super().fit(*a, **k)
    lstm_forecast._OrigSequential = FastSeq


def main():
    build_sharadar()
    fast_models()
    from replay import stage1, weekly, forecasts
    from replay.agent import Replay

    # point-in-time checks
    assert asof.members_live_format(date(2025, 1, 3)) == ["AAA", "BBB", "BRK-B", "CCC"]
    assert "BRK.B" not in asof.sp500_members(asof.sp500_table(), date(2024, 12, 31))
    f = asof.yf_adjusted("AAA", "2018-01-01", "2025-01-04")
    assert f.index[-1] == pd.Timestamp("2025-01-03")
    assert abs(f["Close"].iloc[-1] - asof.unadjusted_close("AAA", date(2025, 1, 3))) < 1e-9
    print("point-in-time checks ok")

    for a in asof.weekly_asof_dates():
        stage1.run_shard(a, 0, 2, config.DATA_DIR / "stage1" / str(a) / "shard_00.csv", limit=2)
        stage1.run_shard(a, 1, 2, config.DATA_DIR / "stage1" / str(a) / "shard_01.csv", limit=2)
        s = stage1.merge(a, config.DATA_DIR / "stage1" / str(a))
        if not s["candidates"]:          # random data rarely clears the gate
            s["candidates"] = asof.members_live_format(a)
            (stage1.out_dir(a) / "stage1.json").write_text(json.dumps(s))
    weekly.run_all()
    tl = pd.read_csv(config.RESULTS_DIR / "whitelist_timeline.csv")
    print(tl)
    if tl["whitelist"].isna().all():
        tl["whitelist"] = "AAA BBB CCC BRK-B"
        tl.to_csv(config.RESULTS_DIR / "whitelist_timeline.csv", index=False)

    print("forecast jobs:", len(forecasts.jobs()))
    forecasts.run_shard(0, 1)
    forecasts.merge()

    for mode in ("copy", "real"):
        Replay(mode, "lstm").run()
    Replay("real", "spy_rsi").run()
    print(f"\nSMOKE OK — outputs in {TMP}")


if __name__ == "__main__":
    main()
