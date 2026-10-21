"""
Factors behind each agent trade, for looking for what the big winners share.

For every distinct trade across the agent books (the same ticker, side and
entry day counted once, real-mode books preferred) it collects:

  funnel      stage-1 LSTM row, stage-2 backtest, sim 1 / sim 2 for the ticker
              in the week it was promoted; weeks carried since promotion
  morning     the LSTM forecast the agent saw on the entry morning
  option      greeks at entry (from the entry mid), spread, DTE, leverage
  alpha/beta  the stock vs SPY over the year (and 60 days) before entry, the
              option's beta (stock beta x leverage)
  peak        the option's best bid (from hold_analysis), the stock and SPY
              moves from entry to that minute, split into the part beta
              explains (market) and the rest (stock-specific)

    python -m replay.trade_factors             # results/analysis/trade_factors.csv
    python -m replay.trade_factors --offline   # skip everything that needs price data
"""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import asof, config, trace
from .agent import implied_vol

R = config.RESULTS_DIR
OUT = R / "analysis"
ONE_MIN = pd.Timedelta(minutes=1)
# Which copy of a trade to keep when several books bought the same thing.
PRIORITY = ["lstm_real_0946_r1001", "lstm_real_0946", "lstm_real", "spy_rsi_real", "lstm_copy"]


def _csv(p) -> pd.DataFrame:
    return trace._csv(p)


# ─────────────────────────────────────────────────────────────────────────────
#  Trades
# ─────────────────────────────────────────────────────────────────────────────

def all_trades() -> pd.DataFrame:
    keep = ["peak_bid", "peak_bid_at", "peak_bid_pct", "peak_mid_pct", "hold_to_expiry_pct",
            "underlying_at_expiry", "ever_positive_bid", "peak_before_exit"]
    frames = []
    for tp in sorted((R / "agent").glob("*/*/trades.csv")):
        variant, book = tp.parent.parent.name, tp.parent.name
        if book not in PRIORITY:
            continue
        t = _csv(tp)
        if t.empty:
            continue
        t["_k"] = pd.to_datetime(t["entry_ts"]).dt.floor("s")
        h = _csv(OUT / f"hold_{variant}_{book}.csv")
        if not h.empty:
            h["_k"] = pd.to_datetime(h["entry"]).dt.floor("s")
            t = t.merge(h[["ticker", "_k"] + [c for c in keep if c in h.columns]],
                        on=["ticker", "_k"], how="left")
        t["variant"], t["book"] = variant, book
        frames.append(t.drop(columns="_k"))
    df = pd.concat(frames, ignore_index=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["entry_date"] = df["entry_ts"].dt.date
    df["_p"] = df["book"].map(PRIORITY.index)
    df = df.sort_values(["_p", "variant", "entry_ts"])
    df["_vb"] = df["variant"] + "/" + df["book"]
    seen = (df.groupby(["ticker", "opt_type", "entry_date"])["_vb"].agg(" ".join)
            .rename("seen_in").reset_index())
    df = df.drop(columns="_vb")
    df = df.drop_duplicates(["ticker", "opt_type", "entry_date"], keep="first").drop(columns="_p")
    return df.merge(seen, on=["ticker", "opt_type", "entry_date"]).sort_values("entry_ts")


# ─────────────────────────────────────────────────────────────────────────────
#  Funnel and morning
# ─────────────────────────────────────────────────────────────────────────────

def funnel_features(variant: str, ticker: str, d: date) -> dict:
    wk = trace.decision_week(d)
    pw = trace.promotion_week(variant, ticker, wk) if wk else None
    out = {"decision_week": wk, "promotion_week": pw,
           "weeks_since_promotion": len([a for a in asof.weekly_asof_dates() if pw and pw < a <= wk])
           if wk and pw else None}
    if pw is None:
        return out
    W = R / "weekly" / pw.isoformat()
    V = R / "variants" / variant / "weekly" / pw.isoformat()

    cand = _csv(W / "lstm_candidates.csv")
    c = cand[cand["ticker"] == ticker] if len(cand) else cand
    if len(c):
        for k in ["rank", "dir_accuracy_pct", "hit_rate_pct", "avg_mae_pct", "avg_snr",
                  "avg_pred_move_pct", "avg_act_move_pct", "dir_pval", "score"]:
            out[f"s1_{k}"] = c[k].iloc[0]
        out["s1_n_candidates"] = len(cand)
    acc = _csv(W / "lstm_accuracy_results.csv")
    a = acc[acc["ticker"] == ticker] if len(acc) else acc
    if len(a):
        out["s1_mae_pctile_all_sp500"] = round((acc["avg_mae_pct"] <= a["avg_mae_pct"].iloc[0]).mean() * 100, 1)
        out["s1_mae_pctile_candidates"] = round((cand["avg_mae_pct"] <= a["avg_mae_pct"].iloc[0]).mean() * 100, 1) \
            if len(cand) else None

    gate = _csv(W / "stage2_gate.csv")
    g = gate[gate["ticker"] == ticker] if len(gate) else gate
    if len(g):
        out.update({"s2_trades": g["backtest_trades"].iloc[0], "s2_pnl": g["backtest_pnl"].iloc[0],
                    "s2_gain_pct_of_stake": g["backtest_gain_pct_of_stake"].iloc[0],
                    "s2_win_pct": g["backtest_win_%"].iloc[0]})
    bt = _csv(W / "intraday_backtest_results.csv")
    b = bt[bt["ticker"] == ticker] if len(bt) else bt
    if len(b):
        out.update({"s2_avg_trailing_pct": b["trailing_pct"].mean(), "s2_avg_eod_pct": b["eod_pct"].mean(),
                    "s2_avg_oracle_pct": b["oracle_pct"].mean(), "s2_max_oracle_pct": b["oracle_pct"].max(),
                    "s2_avg_entry_delta": b["entry_delta"].mean()})

    for tag, where in (("sim1", W), ("sim2", V)):
        tr = _csv(where / f"{tag}_closed_trades.csv")
        x = tr[tr["ticker"] == ticker] if len(tr) else tr
        out[f"{tag}_trades"] = len(x)
        if len(x):
            out.update({f"{tag}_pnl": x["pnl_dollar"].sum(), f"{tag}_win_pct": (x["pnl_dollar"] > 0).mean() * 100,
                        f"{tag}_avg_pnl_pct": x["pnl_pct"].mean(), f"{tag}_avg_peak_pct": x["peak_pnl_pct"].mean(),
                        f"{tag}_max_peak_pct": x["peak_pnl_pct"].max(),
                        f"{tag}_avg_iv_pct": x["iv"].mean() * 100})
    return out


_FC = None


def morning_forecast(ticker: str, d: date, prev_close: float | None) -> dict:
    global _FC
    if _FC is None:
        _FC = _csv(R / "agent" / "lstm_forecasts.csv")
    r = _FC[(_FC["date"] == d.isoformat()) & (_FC["ticker"] == ticker)] if len(_FC) else _FC
    if not len(r) or not prev_close:
        return {}
    r = r.iloc[0]
    return {"fc_1d_pct": (r["tomorrow_price"] / prev_close - 1) * 100,
            "fc_5d_pct": (r["fiveday_price"] / prev_close - 1) * 100,
            "fc_mae_pct": r["daily_mae"] / prev_close * 100,
            "fc_dir_accuracy": r["directional_accuracy"]}


# ─────────────────────────────────────────────────────────────────────────────
#  Option at entry
# ─────────────────────────────────────────────────────────────────────────────

def option_features(t: pd.Series) -> dict:
    call = str(t["opt_type"]).upper().startswith("C")
    S, K, mid = float(t["entry_stock"]), float(t["strike"]), float(t["entry_mid"])
    exp = pd.Timestamp(f"{t['expiry']} 16:00")
    T = max((exp - t["entry_ts"]).total_seconds() / (365 * 86400), 1 / 365)
    r = config.RISK_FREE
    out = {"dte": (pd.Timestamp(t["expiry"]).date() - t["entry_date"]).days,
           "spread_pct_at_entry": 2 * (float(t["entry_ask"]) - mid) / mid * 100,
           "otm_pct": ((K / S - 1) if call else (1 - K / S)) * 100,
           "premium_pct_of_stock": mid / S * 100}
    iv = implied_vol(mid, S, K, T, r, call)
    if iv is None:
        return out
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    delta = norm.cdf(d1) if call else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * iv * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * iv / (2 * math.sqrt(T))
             + (-r * K * math.exp(-r * T) * norm.cdf(d2) if call else r * K * math.exp(-r * T) * norm.cdf(-d2))) / 365
    out.update({"iv_pct": iv * 100, "delta": delta, "gamma": gamma,
                "gamma_pct_per_1pct_move": gamma * (0.01 * S) ** 2 / 2 / mid * 100,
                "theta_pct_per_day": theta / mid * 100, "vega_pct_per_vol_pt": vega / mid * 100,
                "leverage": delta * S / mid,                       # elasticity, signed
                "expected_move_to_expiry_pct": iv * math.sqrt(T) * 100})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Alpha / beta and the move to the peak (need Sharadar + Databento)
# ─────────────────────────────────────────────────────────────────────────────

def _rets(ticker: str, end: date) -> pd.Series:
    df = asof._raw(ticker)
    s = df.loc[df.index <= pd.Timestamp(end), "closeadj"].astype(float)
    return s.pct_change().dropna()


def _ols(y: pd.Series, x: pd.Series) -> tuple[float, float, float, pd.Series]:
    b = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    a = float(y.mean() - b * x.mean())
    resid = y - (a + b * x)
    r2 = float(np.corrcoef(y, x)[0, 1] ** 2)
    return a, b, r2, resid


def alpha_beta(ticker: str, d: date, sign: int) -> dict:
    end = asof.prev_trading_day(d)
    df = pd.concat([_rets(ticker, end).rename("s"), _rets("SPY", end).rename("m")],
                   axis=1, join="inner").dropna()
    out = {}
    for n, tag in ((252, "1y"), (60, "60d")):
        w = df.iloc[-n:]
        if len(w) < n * 0.8:
            continue
        a, b, r2, resid = _ols(w["s"], w["m"])
        out.update({f"beta_{tag}": b, f"alpha_{tag}_ann_pct": a * 252 * 100, f"r2_{tag}": r2,
                    f"idio_vol_{tag}_pct": resid.std() * math.sqrt(252) * 100})
    w = df.iloc[-252:]
    out.update({"vol_1y_pct": w["s"].std() * math.sqrt(252) * 100,
                "vol_20d_pct": df["s"].iloc[-20:].std() * math.sqrt(252) * 100,
                "spy_vol_20d_pct": df["m"].iloc[-20:].std() * math.sqrt(252) * 100,
                "stock_ret_20d_pct": (np.prod(1 + df["s"].iloc[-20:]) - 1) * 100,
                "spy_ret_20d_pct": (np.prod(1 + df["m"].iloc[-20:]) - 1) * 100})
    # positive = the stock's own drift / recent run pointed the way of the trade
    if "alpha_1y_ann_pct" in out:
        out["alpha_1y_with_trade"] = sign * out["alpha_1y_ann_pct"]
    if "alpha_60d_ann_pct" in out:
        out["alpha_60d_with_trade"] = sign * out["alpha_60d_ann_pct"]
    out["ret_20d_with_trade"] = sign * out["stock_ret_20d_pct"]
    return out


def px_at(ticker: str, ts: pd.Timestamp) -> tuple[float | None, str]:
    from . import dbn
    try:
        bars = dbn.equity_bars(ticker, ts.date(), purpose="factor_bars")
    except Exception as e:                       # budget or API trouble: daily fallback
        print(f"[factors] bars {ticker} {ts.date()}: {type(e).__name__}", flush=True)
        bars = pd.DataFrame()
    if not bars.empty:
        done = bars[bars.index + ONE_MIN <= ts]
        if not done.empty:
            return float(done["close"].iloc[-1]), "1m"
        return float(bars["open"].iloc[0]), "1m-open"
    return asof.unadjusted_close(ticker, ts.date()), "daily-close"


def _unadj_hilo(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = asof._raw(ticker)
    w = df[(df.index > pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    k = w["closeunadj"] / w["close"]
    return pd.DataFrame({"high": w["high"] * k, "low": w["low"] * k})


def peak_features(t: pd.Series, f: dict, sign: int) -> dict:
    out = {}
    entry, S0 = t["entry_ts"], float(t["entry_stock"])
    exp = pd.Timestamp(t["expiry"]).date()
    spy0, src0 = px_at("SPY", entry)
    if pd.notna(t.get("peak_bid_at")):
        pk = pd.Timestamp(t["peak_bid_at"])
        S1, src1 = px_at(t["ticker"], pk)
        spy1, _ = px_at("SPY", pk)
        if S1 and spy0 and spy1:
            stock = (S1 / S0 - 1) * 100
            spy = (spy1 / spy0 - 1) * 100
            beta = f.get("beta_1y")
            out.update({"days_to_peak": int(np.busday_count(t["entry_date"], pk.date())),
                        "stock_move_to_peak_pct": stock, "stock_move_to_peak_with_trade": sign * stock,
                        "spy_move_to_peak_pct": spy, "price_source": f"{src0}/{src1}"})
            if beta is not None:
                mkt = sign * beta * spy
                out.update({"market_part_with_trade": mkt,
                            "stock_specific_part_with_trade": sign * stock - mkt})
            lev = f.get("leverage")
            if lev is not None and beta is not None:
                ob = beta * lev
                out["option_beta"] = ob
                out["option_alpha_to_peak_pct"] = float(t["peak_bid_pct"]) - ob * spy
    # Max favourable stock move from entry to the expiry close.
    from . import dbn
    try:
        bars = dbn.equity_bars(t["ticker"], t["entry_date"], purpose="factor_bars")
        after = bars[bars.index >= entry.floor("min")]
        hi = [after["high"].max()] if len(after) else []
        lo = [after["low"].min()] if len(after) else []
    except Exception:
        hi, lo = [], []
    later = _unadj_hilo(t["ticker"], t["entry_date"], exp)
    hi += later["high"].tolist()
    lo += later["low"].tolist()
    if hi and lo:
        out["max_fav_move_to_expiry_pct"] = ((max(hi) / S0 - 1) if sign > 0 else (1 - min(lo) / S0)) * 100
        out["max_adverse_move_to_expiry_pct"] = ((1 - min(lo) / S0) if sign > 0 else (max(hi) / S0 - 1)) * 100
    ends = [e for e in asof.earnings_dates(t["ticker"]) if t["entry_date"] <= e <= exp]
    out["earnings_in_window"] = ends[0].isoformat() if ends else None
    if ends and pd.notna(t.get("peak_bid_at")):
        out["earnings_before_peak"] = ends[0] <= pd.Timestamp(t["peak_bid_at"]).date()
    return out


# ─────────────────────────────────────────────────────────────────────────────

def build(offline: bool = False) -> pd.DataFrame:
    rows = []
    for _, t in all_trades().iterrows():
        sign = 1 if str(t["opt_type"]).upper().startswith("C") else -1
        row = {k: t.get(k) for k in ["seen_in", "variant", "book", "ticker", "direction", "opt_type", "strike",
                                      "expiry", "entry_ts", "entry_stock", "option_cost", "entry_mid",
                                      "contracts", "total_spent", "entry_delta", "snr", "tom_pct",
                                      "rank_position", "n_candidates", "exit_ts", "exit_reason",
                                      "realized_pnl", "peak_pnl_pct", "peak_bid", "peak_bid_at",
                                      "peak_bid_pct", "peak_mid_pct", "hold_to_expiry_pct",
                                      "ever_positive_bid", "peak_before_exit"]}
        row["realized_pct"] = float(t["realized_pnl"]) / float(t["total_spent"]) * 100 \
            if pd.notna(t.get("realized_pnl")) else None
        row.update(funnel_features(t["variant"], t["ticker"], t["entry_date"]))
        row.update(option_features(t))
        if not offline:
            prev = asof.unadjusted_close(t["ticker"], asof.prev_trading_day(t["entry_date"]))
            row.update(morning_forecast(t["ticker"], t["entry_date"], prev))
            row.update(alpha_beta(t["ticker"], t["entry_date"], sign))
            row.update(peak_features(t, row, sign))
        rows.append(row)
        print(f"[factors] {t['ticker']} {t['opt_type']} {t['entry_ts']} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    df = build(a.offline)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "trade_factors_offline.csv" if a.offline else "trade_factors.csv"
    df.to_csv(OUT / name, index=False)
    print(f"::notice title=Trade factors::{len(df)} distinct trades -> results/analysis/{name}", flush=True)
    if not a.offline:
        from . import dbn
        print(f"::notice title=Databento spend::${dbn.ledger()['spent_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
