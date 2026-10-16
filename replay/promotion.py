"""
Stage 4 — promotion. Same rule as app/pipeline/core.py (backtest_pnl_by_ticker,
sim_pnl_by_ticker, compute_promotion), taking the frames directly instead of
reading a run directory.
"""

from __future__ import annotations

import pandas as pd


def backtest_pnl_by_ticker(df) -> dict:
    out = {}
    if df is None or len(df) == 0:
        return out
    for tk, sub in df.groupby("ticker"):
        pnl = float((sub["total_cost"] * sub["trailing_pct"] / 100.0).sum())
        wins = int((sub["trailing_res"] == "Win").sum())
        out[str(tk)] = {
            "pnl": round(pnl, 2),
            "trades": int(len(sub)),
            "win_rate": round(wins / len(sub) * 100, 1) if len(sub) else 0.0,
            "avg_pct": round(float(sub["trailing_pct"].mean()), 2),
        }
    return out


def sim_pnl_by_ticker(metrics: dict | None) -> dict:
    out = {}
    if not metrics:
        return out
    per_ticker = metrics.get("per_ticker_pnl") or metrics.get("per_ticker") or {}
    for tk, stats in per_ticker.items():
        out[str(tk)] = {
            "pnl": round(float(stats.get("pnl", 0.0)), 2),
            "trades": int(stats.get("trades", 0)),
            "win_rate": float(stats.get("win_rate", 0.0)),
        }
    return out


def compute_promotion(candidates: list[str], bt_df, sim_metrics: dict | None,
                      min_trades: int = 1):
    bt = backtest_pnl_by_ticker(bt_df)
    sm = sim_pnl_by_ticker(sim_metrics)
    universe = candidates or sorted(set(bt) | set(sm))

    rows = []
    for tk in universe:
        b, s = bt.get(tk), sm.get(tk)
        b_trades = b["trades"] if b else 0
        s_trades = s["trades"] if s else 0
        b_pnl = b["pnl"] if b else 0.0
        s_pnl = s["pnl"] if s else 0.0
        if b_trades < min_trades and s_trades < min_trades:
            status = "no trades (neither)"
        elif b_trades < min_trades:
            status = "no trades (backtest)"
        elif s_trades < min_trades:
            status = "no trades (sim)"
        elif b_pnl > 0 and s_pnl > 0:
            status = "promoted"
        elif b_pnl > 0:
            status = "sim negative"
        elif s_pnl > 0:
            status = "backtest negative"
        else:
            status = "negative in both"
        rows.append({
            "ticker": tk, "backtest_pnl": b_pnl, "backtest_trades": b_trades,
            "backtest_win_%": b["win_rate"] if b else 0.0,
            "sim_pnl": s_pnl, "sim_trades": s_trades,
            "sim_win_%": s["win_rate"] if s else 0.0, "status": status,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["status", "sim_pnl"],
            key=lambda c: c.map({"promoted": 0}).fillna(1) if c.name == "status" else c,
            ascending=[True, False],
        ).reset_index(drop=True)
    promoted = df.loc[df["status"] == "promoted", "ticker"].tolist() if not df.empty else []
    return df, promoted


def stage2_gate(candidates: list[str], bt_df, stake: float, min_gain_pct: float,
                min_trades: int = 1):
    """Tickers whose total stage-2 P&L >= min_gain_pct % of the per-trade stake."""
    bt = backtest_pnl_by_ticker(bt_df)
    rows, passed = [], []
    for tk in candidates:
        b = bt.get(tk)
        trades = b["trades"] if b else 0
        pnl = b["pnl"] if b else 0.0
        gain = pnl / stake * 100 if stake else 0.0
        ok = trades >= min_trades and pnl > 0 and gain >= min_gain_pct
        rows.append({"ticker": tk, "backtest_trades": trades, "backtest_pnl": pnl,
                     "backtest_gain_pct_of_stake": round(gain, 1),
                     "backtest_win_%": b["win_rate"] if b else 0.0,
                     "passed_gate": ok})
        if ok:
            passed.append(tk)
    return pd.DataFrame(rows), passed


def sim_survivors(tickers: list[str], metrics: dict | None, min_trades: int = 1):
    sm = sim_pnl_by_ticker(metrics)
    out = {}
    for tk in tickers:
        s = sm.get(tk)
        out[tk] = {"trades": s["trades"] if s else 0, "pnl": s["pnl"] if s else 0.0,
                   "win_rate": s["win_rate"] if s else 0.0}
    positive = [t for t, v in out.items() if v["trades"] >= min_trades and v["pnl"] > 0]
    return out, positive


def funnel_table(gate_df, sim1: dict, sim2: dict, stake_pct: float) -> pd.DataFrame:
    df = gate_df.copy()
    df["sim1_trades"] = df["ticker"].map(lambda t: sim1.get(t, {}).get("trades"))
    df["sim1_pnl"] = df["ticker"].map(lambda t: sim1.get(t, {}).get("pnl"))
    df["sim2_trades"] = df["ticker"].map(lambda t: sim2.get(t, {}).get("trades"))
    df["sim2_pnl"] = df["ticker"].map(lambda t: sim2.get(t, {}).get("pnl"))

    def status(r):
        if r["backtest_trades"] == 0:
            return "no trades (backtest)"
        if not r["passed_gate"]:
            return f"backtest below {stake_pct:g}%"
        if not r["sim1_trades"]:
            return "no trades (sim 1)"
        if r["sim1_pnl"] <= 0:
            return "sim 1 negative"
        if not r["sim2_trades"]:
            return "no trades (sim 2)"
        if r["sim2_pnl"] <= 0:
            return "sim 2 negative"
        return "promoted"
    df["status"] = df.apply(status, axis=1) if len(df) else []
    return df
