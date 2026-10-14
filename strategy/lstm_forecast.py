"""
╔══════════════════════════════════════════════════════════════════════════╗
║         LSTM_FORECAST — shared LSTM module for Prometheus v2/v3         ║
║                                                                          ║
║  Houses the historical data fetcher and the Golden LSTM forecast        ║
║  function. Both v2 and v3 import from here; deleting v2 will not        ║
║  break v3.                                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import yfinance as yf

from dotenv import load_dotenv
load_dotenv()

ET = ZoneInfo("America/New_York")

TRADIER_TOKEN    = os.getenv("TRADIER_TOKEN")
TRADIER_BASE_URL = "https://sandbox.tradier.com/v1"

log = logging.getLogger("lstm_forecast")


def fetch_historical_data(ticker: str, start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Tradier, falling back to yfinance on failure.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume (indexed by date).
    Returns empty DataFrame if both sources fail.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # --- Primary: Tradier historical endpoint ---
    try:
        url = f"{TRADIER_BASE_URL}/markets/history"
        params = {"symbol": ticker, "interval": "daily", "start": start, "end": end}
        headers = {"Authorization": f"Bearer {TRADIER_TOKEN}", "Accept": "application/json"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        history = resp.json().get("history")
        if history and history.get("day"):
            days = history["day"]
            if isinstance(days, dict):
                days = [days]
            df = pd.DataFrame(days)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            }, inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            if len(df) >= 10:
                log.info(f"[{ticker}] Tradier historical: {len(df)} rows loaded")
                return df
            log.warning(f"[{ticker}] Tradier historical returned only {len(df)} rows — falling back to yfinance")
    except Exception as e:
        log.warning(f"[{ticker}] Tradier historical failed ({e}) — falling back to yfinance")

    # --- Fallback: yfinance ---
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        log.info(f"[{ticker}] yfinance fallback: {len(df)} rows loaded")
        return df
    except Exception as e:
        log.warning(f"[{ticker}] yfinance fallback also failed ({e})")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
#  TENSORFLOW IMPORTS — deferred so the base module imports fast
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import gc
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import pandas_market_calendars as mcal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


# ─────────────────────────────────────────────────────────────────────────────
#  GOLDEN LSTM — UNIVERSAL (any ticker)
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm_forecast(ticker: str, current_price: float) -> dict | None:
    """
    Train the Golden LSTM on any ticker and return forecast results.
    Returns None if data is insufficient or training fails.
    """
    try:
        # Clear TF session at the START so each ticker gets a completely fresh graph
        tf.keras.backend.clear_session()
        gc.collect()

        end_date = datetime.today().strftime("%Y-%m-%d")
        data     = fetch_historical_data(ticker, start="2018-01-01", end=end_date)
        market   = fetch_historical_data("SPY",  start="2018-01-01", end=end_date)

        if data.empty or market.empty or len(data) < 150:
            log.debug(f"[{ticker}] Insufficient data: ticker_rows={len(data)}, market_rows={len(market)}")
            return None

        # Verify we got the right ticker data (not SPY accidentally)
        last_close = float(data["Close"].iloc[-1])
        log.info(f"[{ticker}] Data loaded: {len(data)} rows | last_close=${last_close:.2f} | current_price=${current_price:.2f}")

        # Feature engineering
        close  = data["Close"].squeeze()
        high   = data["High"].squeeze()
        low    = data["Low"].squeeze()
        delta  = close.diff()
        gain   = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss   = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        # Guard against division-by-zero when 14-day window has no down days.
        # Without this, RSI = inf, which survives dropna() and poisons MinMaxScaler.
        loss_safe = loss.replace(0, 1e-10)
        data["RSI"]           = 100 - (100 / (1 + gain / loss_safe))
        data["Market_Change"] = market["Close"].squeeze().pct_change()
        # MACD
        ema12            = close.ewm(span=12, adjust=False).mean()
        ema26            = close.ewm(span=26, adjust=False).mean()
        data["MACD"]     = ema12 - ema26
        # Bollinger Bands (20-day)
        bb_mid            = close.rolling(20).mean()
        bb_std            = close.rolling(20).std()
        data["BB_mid"]    = bb_mid
        data["BB_upper"]  = bb_mid + 2 * bb_std
        data["BB_lower"]  = bb_mid - 2 * bb_std
        # ATR (14-day)
        prev_close        = close.shift(1)
        tr                = pd.concat([high - low,
                                        (high - prev_close).abs(),
                                        (low  - prev_close).abs()], axis=1).max(axis=1)
        data["ATR"]       = tr.rolling(14).mean()
        # Volume (log-scaled to reduce magnitude differences)
        data["Volume"]    = np.log1p(data["Volume"].squeeze())

        df = data[["Close", "RSI", "Market_Change", "Volume", "MACD",
                   "BB_upper", "BB_lower", "BB_mid", "ATR"]].dropna()
        if len(df) < 150:
            return None

        # Fit a brand-new scaler on THIS ticker's data only
        train_len   = int(len(df) * 0.8)
        scaler      = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df.values)                      # fit on full dataset so current price is always in range
        scaled      = scaler.transform(df.values)
        log.info(f"[{ticker}] Scaler range: Close min=${scaler.data_min_[0]:.2f} max=${scaler.data_max_[0]:.2f}")
        train_data  = scaled[:train_len]

        SEQ = 90
        x_train, y_train = [], []
        for i in range(SEQ, len(train_data)):
            x_train.append(train_data[i - SEQ:i])
            y_train.append(train_data[i, 0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        N_FEAT = df.shape[1]   # 9 features
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(SEQ, N_FEAT)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x_train, y_train, batch_size=32, epochs=30, verbose=0)

        # Test accuracy
        test_data = scaled[train_len - SEQ:]
        x_test    = np.array([test_data[i - SEQ:i] for i in range(SEQ, len(test_data))])
        preds_sc  = model.predict(x_test, verbose=0)
        dummy     = np.zeros((len(preds_sc), N_FEAT))
        dummy[:, 0] = preds_sc.flatten()
        predictions = scaler.inverse_transform(dummy)[:, 0]
        y_test_real = df["Close"].values[train_len:]
        daily_mae   = mean_absolute_error(y_test_real, predictions)
        mae_pct     = (daily_mae / current_price) * 100   # MAE as % of price

        # Directional accuracy on the test set:
        # For each test day, did the model predict the right direction
        # relative to the previous actual close? This is the metric that
        # matters for entry decisions — magnitude (MAE) matters less than
        # whether we'd have bought a call or a put.
        # We compare each prediction[i] to y_test_real[i-1] (the prior
        # actual close) and check if the predicted direction matches
        # y_test_real[i] vs y_test_real[i-1].
        try:
            if len(predictions) >= 2 and len(y_test_real) >= 2:
                # Align lengths defensively (sometimes one extra row from slicing)
                n = min(len(predictions), len(y_test_real))
                preds_aligned = predictions[:n]
                reals_aligned = y_test_real[:n]
                # Prior actual close = reals_aligned shifted by 1
                prior_real    = reals_aligned[:-1]
                predicted_now = preds_aligned[1:]
                actual_now    = reals_aligned[1:]
                predicted_up  = predicted_now > prior_real
                actual_up     = actual_now    > prior_real
                hits          = (predicted_up == actual_up).sum()
                total         = len(predicted_up)
                directional_accuracy = (hits / total * 100) if total > 0 else 0.0
            else:
                directional_accuracy = 0.0
        except Exception as e:
            log.warning(f"[{ticker}] Directional accuracy calc failed ({e}) — defaulting to 0.0")
            directional_accuracy = 0.0

        log.info(
            f"[{ticker}] Test metrics: MAE={daily_mae:.2f} ({mae_pct:.2f}%) | "
            f"directional_accuracy={directional_accuracy:.1f}%"
        )

        # Tomorrow forecast
        last_seq         = scaled[-SEQ:].reshape(1, SEQ, N_FEAT)
        tom_sc           = model.predict(last_seq, verbose=0)
        raw_pred         = float(tom_sc[0, 0])
        dummy_t          = np.zeros((1, N_FEAT))
        dummy_t[0, 0]    = raw_pred
        tomorrow_price   = float(scaler.inverse_transform(dummy_t)[0, 0])

        log.info(f"[{ticker}] raw_pred={raw_pred:.6f} | scaler_min=${scaler.data_min_[0]:.2f} | scaler_max=${scaler.data_max_[0]:.2f} | descaled=${tomorrow_price:.2f}")

        # Sanity check: forecast should be within 60% of current price
        if current_price > 0 and abs(tomorrow_price - current_price) / current_price > 0.60:
            log.warning(f"[{ticker}] Forecast ${tomorrow_price:.2f} deviates >60% from current ${current_price:.2f} — skipping")
            tf.keras.backend.clear_session()
            del model
            gc.collect()
            return None

        # 5-day recursive forecast
        temp = last_seq.copy()
        for _ in range(5):
            step_sc          = model.predict(temp, verbose=0)[0]
            new_row          = np.zeros((1, 1, N_FEAT))
            new_row[0, 0, 0] = step_sc[0]
            new_row[0, 0, 1:] = temp[0, -1, 1:]          # carry other features forward
            temp             = np.append(temp[:, 1:, :], new_row, axis=1)
        dummy_f       = np.zeros((1, N_FEAT))
        dummy_f[0, 0] = step_sc[0]
        fiveday_price = float(scaler.inverse_transform(dummy_f)[0, 0])

        tom_pct     = ((tomorrow_price - current_price) / current_price) * 100
        fiveday_pct = ((fiveday_price  - current_price) / current_price) * 100

        # Get next trading day (tz-aware to match rest of file)
        nyse = mcal.get_calendar('NYSE')
        _now_et = datetime.now(ET)
        today = _now_et.strftime('%Y-%m-%d')
        schedule = nyse.schedule(start_date=today, end_date=(_now_et + timedelta(days=7)).strftime('%Y-%m-%d'))
        next_trading_day = schedule.index[1].strftime('%m/%d/%y') if len(schedule) > 1 else (_now_et + timedelta(days=1)).strftime('%m/%d/%y')

        # Profitability confidence:
        # Good signal = large predicted move AND MAE is small relative to the move
        signal_to_noise = abs(tom_pct) / max(mae_pct, 0.1)

        result = {
            "ticker":         ticker,
            "current_price":  current_price,
            "tomorrow_price": tomorrow_price,
            "fiveday_price":  fiveday_price,
            "tom_pct":        tom_pct,
            "fiveday_pct":    fiveday_pct,
            "daily_mae":      daily_mae,
            "mae_pct":        mae_pct,
            "signal_to_noise": signal_to_noise,
            "directional_accuracy": directional_accuracy,
            "next_trading_day": next_trading_day,
        }

        # Free GPU/RAM immediately — models accumulate across hundreds of tickers
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return result

    except Exception as e:
        log.exception(f"LSTM failed for {ticker}")
        tf.keras.backend.clear_session()
        gc.collect()
        return None
