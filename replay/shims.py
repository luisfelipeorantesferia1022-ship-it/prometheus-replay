"""
Rebinding the strategy modules onto point-in-time data.

The strategy files in strategy/ are byte-for-byte copies of app/ in the live
repo. Nothing in them is edited; this module swaps their data sources (the
same technique the console's pipeline runners use to inject the whitelist).
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date, timedelta

import pandas as pd

from . import asof, config

if str(config.STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(config.STRATEGY_DIR))


def seed_for(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return (int(h[:8], 16) + config.SEED_BASE) % (2**31 - 1)


class YFShim:
    """Stands in for the `yf` name inside lstm_test (auto_adjust=True path)."""

    def download(self, ticker, start=None, end=None, progress=False,
                 auto_adjust=True, **_):
        if auto_adjust:
            return asof.yf_adjusted(ticker, start, end)
        return asof.split_adjusted(ticker, start, end)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — lstm_test
# ─────────────────────────────────────────────────────────────────────────────

def lstm_test_module(asof_date: date):
    import lstm_test
    import tensorflow as tf

    lstm_test.yf = YFShim()
    lstm_test.get_sp500_tickers = lambda: asof.members_live_format(asof_date)

    if not hasattr(lstm_test, "_orig_train_model"):
        lstm_test._orig_train_model = lstm_test.train_model

    def seeded_train(scaled, train_len):
        tf.keras.utils.set_random_seed(seed_for("stage1", asof_date, lstm_test._current))
        return lstm_test._orig_train_model(scaled, train_len)

    lstm_test.train_model = seeded_train
    lstm_test._current = None
    return lstm_test


def stage1_end_date(asof_date: date) -> str:
    """lstm_test passes datetime.today() as yf `end` (exclusive): the weekend."""
    return (asof_date + timedelta(days=1)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Stages 2 and 3 — portfolio_sim_v3 / intraday_backtest
# ─────────────────────────────────────────────────────────────────────────────

def sim_fetch(anchor: date):
    """portfolio_sim_v3.fetch_historical_data with the same window arithmetic."""
    def fetch_historical_data(ticker, days_back, oos_shift_days=None, cfg=None):
        shift = 0 if oos_shift_days is None else oos_shift_days
        if cfg is not None and oos_shift_days is None:
            shift = cfg.oos_shift_days
        end = anchor - timedelta(days=shift)
        start = end - timedelta(days=days_back + 60)
        df = asof.split_adjusted(ticker, start, end)
        return None if df.empty else df
    return fetch_historical_data


def sim_data(universe: list[str], anchor: date, sim_days: int = 365) -> dict:
    """The dict run_sim(cfg, data=...) accepts, built the way load_all_data would."""
    fetch = sim_fetch(anchor)
    spy = fetch("SPY", sim_days, 0)
    if spy is None:
        raise RuntimeError(f"SPY missing for anchor {anchor}")
    ticker_data, earnings = {}, {}
    for t in universe:
        df = fetch(t, sim_days, 0)
        if df is None:
            continue
        ticker_data[t] = df
        earnings[t] = asof.earnings_dates(t)
    return {"spy_df": spy, "ticker_data": ticker_data, "earnings_data": earnings}


# ─────────────────────────────────────────────────────────────────────────────
#  Agent — lstm_forecast
# ─────────────────────────────────────────────────────────────────────────────

def lstm_forecast_module(data_asof: date, tag: str):
    """
    run_lstm_forecast reads Tradier daily history through the prior close
    (the agent scans at the open). The shim ignores `end` and stops at
    `data_asof`.
    """
    import lstm_forecast
    import tensorflow as tf

    end_excl = data_asof + timedelta(days=1)

    def fetch_historical_data(ticker, start="2018-01-01", end=None):
        return asof.split_adjusted(ticker, start, end_excl)

    lstm_forecast.fetch_historical_data = fetch_historical_data

    if not hasattr(lstm_forecast, "_OrigSequential"):
        lstm_forecast._OrigSequential = lstm_forecast.Sequential

    def seeded_sequential(*a, **k):
        # Seed immediately before the weights are created.
        tf.keras.utils.set_random_seed(seed_for("agent", data_asof, tag, lstm_forecast._current))
        return lstm_forecast._OrigSequential(*a, **k)

    lstm_forecast.Sequential = seeded_sequential
    lstm_forecast._current = None
    return lstm_forecast
