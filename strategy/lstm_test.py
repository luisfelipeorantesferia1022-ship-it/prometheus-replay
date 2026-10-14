"""
lstm_test.py — Standalone LSTM accuracy tester for S&P 500
Tests the same architecture as paper_trading.py / trading_agent_v2.py
against the last 30 trading days of historical data per ticker.

Outputs: lstm_accuracy_results.csv (sorted by directional accuracy desc)
         Console summary at the end
"""

import gc
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import tensorflow as tf
from io import StringIO
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_START   = "2018-01-01"
SEQ           = 90          # lookback window (same as production)
EVAL_DAYS     = 30          # rolling days to evaluate per ticker
HIT_THRESHOLD = 0.02        # 2% — price within this % = "hit"
OUTPUT_CSV    = "lstm_accuracy_results.csv"
BATCH_SIZE    = 50          # Clear RAM and restart after this many tickers

# ── S&P 500 ticker loader ────────────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15,
        )
        resp.raise_for_status()
        tables  = pd.read_html(StringIO(resp.text))
        tickers = [t.replace(".", "-") for t in tables[0]["Symbol"].tolist()]
        print(f"Loaded {len(tickers)} tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"Wikipedia failed ({e}) — trying GitHub backup …")

    try:
        url     = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df      = pd.read_csv(url)
        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
        print(f"Loaded {len(tickers)} tickers from GitHub backup.")
        return tickers
    except Exception as e:
        print(f"GitHub backup failed: {e}. Using short fallback list.")
        return [
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","UNH","JPM",
            "V","XOM","JNJ","WMT","MA","PG","HD","CVX","MRK","ABBV",
        ]

# ── Feature engineering (identical to production) ───────────────────────────
def build_features(data: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    if isinstance(market.columns, pd.MultiIndex):
        market.columns = [col[0] for col in market.columns]

    data  = data.copy()
    close = data["Close"].squeeze()
    high  = data["High"].squeeze()
    low   = data["Low"].squeeze()
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
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
    # Volume (log-scaled)
    data["Volume"]    = np.log1p(data["Volume"].squeeze())

    return data[["Close", "RSI", "Market_Change", "Volume", "MACD",
                 "BB_upper", "BB_lower", "BB_mid", "ATR"]].dropna()

# ── Build and train model (identical architecture to production) ─────────────
def train_model(scaled: np.ndarray, train_len: int):
    train_data = scaled[:train_len]
    x_train, y_train = [], []
    for i in range(SEQ, len(train_data)):
        x_train.append(train_data[i - SEQ:i])
        y_train.append(train_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    N_FEAT = scaled.shape[1]   # 9 features
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
    return model

# ── Evaluate one ticker ──────────────────────────────────────────────────────
def evaluate_ticker(ticker: str, end_date: str) -> dict | None:
    try:
        tf.keras.backend.clear_session()
        gc.collect()

        data   = yf.download(ticker, start=TRAIN_START, end=end_date, progress=False, auto_adjust=True)
        market = yf.download("SPY",  start=TRAIN_START, end=end_date, progress=False, auto_adjust=True)

        if data.empty or market.empty or len(data) < 150 + EVAL_DAYS:
            return None

        df = build_features(data, market)
        if len(df) < 150 + EVAL_DAYS:
            return None

        # We need at least SEQ + EVAL_DAYS rows after train split
        # Train on everything except the last EVAL_DAYS rows
        eval_cutoff = len(df) - EVAL_DAYS
        df_train    = df.iloc[:eval_cutoff]

        scaler     = MinMaxScaler(feature_range=(0, 1))
        scaled_all = scaler.fit_transform(df.values)

        train_len  = int(eval_cutoff * 0.8)
        scaled_tr  = scaled_all[:eval_cutoff]

        model = train_model(scaled_tr, train_len)

        # Predict one step ahead for each of the last EVAL_DAYS days
        dir_correct  = []
        abs_errors   = []
        pct_pred_moves = []
        pct_act_moves  = []
        hits           = []

        closes = df["Close"].values

        for i in range(EVAL_DAYS):
            # Window ends at index eval_cutoff + i - 1 (last known bar before prediction target)
            window_end   = eval_cutoff + i      # exclusive — the row we're predicting
            window_start = window_end - SEQ

            if window_start < 0:
                continue

            N_FEAT     = scaled_all.shape[1]
            seq_scaled = scaled_all[window_start:window_end].reshape(1, SEQ, N_FEAT)

            pred_sc  = model.predict(seq_scaled, verbose=0)
            dummy    = np.zeros((1, N_FEAT))
            dummy[0, 0] = float(pred_sc[0, 0])
            pred_price  = float(scaler.inverse_transform(dummy)[0, 0])

            actual_price = float(closes[window_end])   # next-day actual close
            prev_price   = float(closes[window_end - 1])

            pred_dir   = 1 if pred_price > prev_price else -1
            actual_dir = 1 if actual_price > prev_price else -1
            dir_correct.append(1 if pred_dir == actual_dir else 0)

            abs_errors.append(abs(pred_price - actual_price))

            pct_pred  = (pred_price  - prev_price) / prev_price * 100
            pct_act   = (actual_price - prev_price) / prev_price * 100
            pct_pred_moves.append(pct_pred)
            pct_act_moves.append(pct_act)

            hit = abs(pred_price - actual_price) / actual_price <= HIT_THRESHOLD
            hits.append(1 if hit else 0)

        if not dir_correct:
            return None

        avg_mae     = float(np.mean(abs_errors))
        avg_price   = float(np.mean(closes[eval_cutoff:eval_cutoff + EVAL_DAYS]))
        mae_pct     = avg_mae / avg_price * 100
        avg_pred_mv = float(np.mean(np.abs(pct_pred_moves)))
        snr         = avg_pred_mv / max(mae_pct, 0.01)

        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return {
            "ticker":            ticker,
            "dir_accuracy_pct":  round(float(np.mean(dir_correct)) * 100, 1),
            "avg_mae":           round(avg_mae, 4),
            "avg_mae_pct":       round(mae_pct, 3),
            "avg_snr":           round(snr, 3),
            "avg_pred_move_pct": round(float(np.mean(pct_pred_moves)), 3),
            "avg_act_move_pct":  round(float(np.mean(pct_act_moves)), 3),
            "hit_rate_pct":      round(float(np.mean(hits)) * 100, 1),
            "n_days_evaluated":  len(dir_correct),
        }

    except Exception as e:
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"  ERROR {ticker}: {e}")
        return None

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    tickers  = get_sp500_tickers()
    total    = len(tickers)
    end_date = datetime.today().strftime("%Y-%m-%d")
    failed   = []

    _COLUMNS = ["ticker", "dir_accuracy_pct", "avg_mae", "avg_mae_pct", "avg_snr",
                "avg_pred_move_pct", "avg_act_move_pct", "hit_rate_pct", "n_days_evaluated"]

    # Load already-completed tickers from CSV so we can resume after a crash or batch restart
    completed = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            _existing = pd.read_csv(OUTPUT_CSV)
            completed = set(_existing["ticker"].tolist())
            print(f"Resuming — {len(completed)} tickers already done, skipping.")
        except Exception:
            completed = set()
    else:
        with open(OUTPUT_CSV, "w", newline="") as _f:
            _f.write(",".join(_COLUMNS) + "\n")

    remaining = [t for t in tickers if t not in completed]
    print(f"\nEvaluating {len(remaining)} tickers | eval window: last {EVAL_DAYS} trading days each\n")

    for idx, ticker in enumerate(remaining, start=len(completed) + 1):
        print(f"[{idx}/{total}] {ticker} ...", end=" ", flush=True)
        result = evaluate_ticker(ticker, end_date)
        if result:
            with open(OUTPUT_CSV, "a", newline="") as _f:
                _f.write(",".join(str(result[c]) for c in _COLUMNS) + "\n")
            print(f"dir={result['dir_accuracy_pct']}%  mae={result['avg_mae_pct']:.2f}%  "
                  f"snr={result['avg_snr']:.2f}  hit={result['hit_rate_pct']}%")
        else:
            failed.append(ticker)
            print("SKIPPED (insufficient data or error)")

        # Flush RAM every BATCH_SIZE tickers
        if (idx - len(completed)) % BATCH_SIZE == 0:
            print(f"\n--- Batch flush at ticker {idx} — clearing RAM ---\n")
            gc.collect()
            tf.keras.backend.clear_session()

    # ── Load completed results from the checkpoint CSV (source of truth) ──────
    # NOTE: results are written to OUTPUT_CSV incrementally inside the loop above;
    # there is no in-memory `results` list, so we read the file back here.
    try:
        df_out = pd.read_csv(OUTPUT_CSV)
    except Exception:
        df_out = pd.DataFrame()

    if df_out.empty:
        print("\nNo results to export.")
        return

    # Re-sort the raw results CSV by directional accuracy descending
    df_out = df_out.sort_values("dir_accuracy_pct", ascending=False)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults sorted and exported to {OUTPUT_CSV}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL SUMMARY")
    print("=" * 55)
    print(f"  Tickers evaluated:         {len(df_out)}")
    print(f"  Tickers skipped/failed:    {len(failed)}")
    print(f"  Overall dir accuracy:      {df_out['dir_accuracy_pct'].mean():.1f}%")
    print(f"  Average MAE (price):       {df_out['avg_mae'].mean():.4f}")
    print(f"  Average MAE (%):           {df_out['avg_mae_pct'].mean():.3f}%")
    print(f"  Average SNR:               {df_out['avg_snr'].mean():.3f}")
    print(f"  Average hit rate (±2%):    {df_out['hit_rate_pct'].mean():.1f}%")
    print("=" * 55)

    top5 = df_out.head(5)[["ticker", "dir_accuracy_pct", "avg_snr", "hit_rate_pct"]]
    print("\nTop 5 by directional accuracy:")
    print(top5.to_string(index=False))

    bot5 = df_out.tail(5)[["ticker", "dir_accuracy_pct", "avg_snr", "hit_rate_pct"]]
    print("\nBottom 5 by directional accuracy:")
    print(bot5.to_string(index=False))

    # ── Candidate qualification (HARD filters + composite rank) ───────────────
    # Three gates, ALL must pass:
    #   1. Directional accuracy beats a coin flip at p < SIG_ALPHA (one-sided binomial)
    #   2. Magnitude is usable: hit_rate_pct >= MIN_HIT_PCT
    #   3. Error is tight:      avg_mae_pct <= universe median
    # Survivors ranked by composite z-score: direction + hit-rate, minus MAE%.
    # Only survivors are written, to a separate CANDIDATES_CSV (backtest handoff).
    from math import comb

    SIG_ALPHA            = 0.05    # directional significance threshold
    MIN_HIT_PCT          = 50.0    # magnitude-calibration floor
    W_DIR, W_HIT, W_MAE  = 0.35, 0.35, 0.30
    CANDIDATES_CSV       = "lstm_candidates.csv"

    def _binom_p(k: int, n: int, p: float = 0.5) -> float:
        # one-sided P(X >= k) for n Bernoulli(p) trials
        k = max(0, min(k, n))
        return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))

    try:
        cand = df_out.copy()
        cand["k_correct"] = (
            cand["dir_accuracy_pct"] / 100.0 * cand["n_days_evaluated"]
        ).round().astype(int)
        cand["dir_pval"] = [
            _binom_p(int(k), int(n))
            for k, n in zip(cand["k_correct"], cand["n_days_evaluated"])
        ]

        # Composite z-score against the FULL universe (how far above the field).
        def _z(series):
            sd = series.std(ddof=0)
            return (series - series.mean()) / sd if sd else series * 0.0

        cand["score"] = (
            W_DIR * _z(cand["dir_accuracy_pct"]) +
            W_HIT * _z(cand["hit_rate_pct"]) -
            W_MAE * _z(cand["avg_mae_pct"])
        )

        mae_cap = cand["avg_mae_pct"].median()
        gate = (
            (cand["dir_pval"] < SIG_ALPHA) &
            (cand["hit_rate_pct"] >= MIN_HIT_PCT) &
            (cand["avg_mae_pct"] <= mae_cap)
        )
        survivors = cand[gate].sort_values("score", ascending=False).copy()
        survivors.insert(0, "rank", range(1, len(survivors) + 1))

        out_cols = ["rank", "ticker", "dir_accuracy_pct", "hit_rate_pct",
                    "avg_mae_pct", "avg_snr", "avg_pred_move_pct",
                    "avg_act_move_pct", "dir_pval", "score"]
        survivors[out_cols].to_csv(CANDIDATES_CSV, index=False)

        print("\n" + "=" * 72)
        print(f"  RANKED CANDIDATES (p<{SIG_ALPHA} | hit>={MIN_HIT_PCT:.0f}% | mae%<=median {mae_cap:.2f})")
        print("=" * 72)
        print(f"  {len(survivors)} of {len(cand)} tickers cleared all three gates")
        print(f"  Written to {CANDIDATES_CSV}\n")
        if survivors.empty:
            print("  (no survivors — loosen gates or inspect the universe)")
        else:
            hdr = f"  {'#':<4}{'TICK':<7}{'DIR%':>6}{'HIT%':>6}{'MAE%':>7}{'SNR':>6}{'PVAL':>7}{'SCORE':>8}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for _, r in survivors.iterrows():
                print(f"  {int(r['rank']):<4}{r['ticker']:<7}{r['dir_accuracy_pct']:>6.1f}"
                      f"{r['hit_rate_pct']:>6.1f}{r['avg_mae_pct']:>7.2f}{r['avg_snr']:>6.2f}"
                      f"{r['dir_pval']:>7.3f}{r['score']:>8.2f}")
            print()
            print(f"  Candidate tickers: {survivors['ticker'].tolist()}")
        print("=" * 72)
    except Exception as e:
        print(f"\nCould not compute ranked candidates: {e}")

if __name__ == "__main__":
    main()
