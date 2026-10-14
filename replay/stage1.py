"""
Stage 1 — the LSTM sweep, as it would have run on the weekend after `asof`.

    python -m replay.stage1 shard --asof 2025-01-03 --shard 0 --nshards 6
    python -m replay.stage1 merge --asof 2025-01-03

`shard` evaluates every k-th S&P 500 member (membership as of that date) with
lstm_test.evaluate_ticker, unmodified, on point-in-time data. `merge` stitches
the shards and applies the candidate gate exactly as lstm_test.main() does,
then the console's "Cap candidates" (top N by score).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from math import comb
from pathlib import Path

import pandas as pd

from . import asof, config, shims

COLUMNS = ["ticker", "dir_accuracy_pct", "avg_mae", "avg_mae_pct", "avg_snr",
           "avg_pred_move_pct", "avg_act_move_pct", "hit_rate_pct", "n_days_evaluated"]


def out_dir(asof_date: date) -> Path:
    return config.RESULTS_DIR / "weekly" / asof_date.isoformat()


def run_shard(asof_date: date, shard: int, nshards: int, out: Path,
              limit: int | None = None) -> pd.DataFrame:
    lt = shims.lstm_test_module(asof_date)
    tickers = lt.get_sp500_tickers()
    mine = [t for i, t in enumerate(tickers) if i % nshards == shard]
    if limit:
        mine = mine[:limit]
    end_date = shims.stage1_end_date(asof_date)
    print(f"[stage1] {asof_date} shard {shard}/{nshards}: {len(mine)} of "
          f"{len(tickers)} members", flush=True)
    rows, failed = [], []
    for i, t in enumerate(mine, 1):
        t0 = time.time()
        lt._current = t
        r = lt.evaluate_ticker(t, end_date)
        if r:
            rows.append(r)
        else:
            failed.append(t)
        print(f"  [{i}/{len(mine)}] {t} {'ok' if r else 'SKIPPED'} "
              f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    Path(str(out) + ".failed").write_text("\n".join(failed))
    return df


def rank_candidates(df_out: pd.DataFrame) -> pd.DataFrame:
    """Verbatim logic of the candidate block at the end of lstm_test.main()."""
    SIG_ALPHA = 0.05
    MIN_HIT_PCT = 50.0
    W_DIR, W_HIT, W_MAE = 0.35, 0.35, 0.30

    def _binom_p(k: int, n: int, p: float = 0.5) -> float:
        k = max(0, min(k, n))
        return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))

    cand = df_out.copy()
    cand["k_correct"] = (cand["dir_accuracy_pct"] / 100.0 * cand["n_days_evaluated"]).round().astype(int)
    cand["dir_pval"] = [_binom_p(int(k), int(n))
                        for k, n in zip(cand["k_correct"], cand["n_days_evaluated"])]

    def _z(series):
        sd = series.std(ddof=0)
        return (series - series.mean()) / sd if sd else series * 0.0

    cand["score"] = (W_DIR * _z(cand["dir_accuracy_pct"]) +
                     W_HIT * _z(cand["hit_rate_pct"]) -
                     W_MAE * _z(cand["avg_mae_pct"]))
    mae_cap = cand["avg_mae_pct"].median()
    gate = ((cand["dir_pval"] < SIG_ALPHA) &
            (cand["hit_rate_pct"] >= MIN_HIT_PCT) &
            (cand["avg_mae_pct"] <= mae_cap))
    survivors = cand[gate].sort_values("score", ascending=False).copy()
    survivors.insert(0, "rank", range(1, len(survivors) + 1))
    out_cols = ["rank", "ticker", "dir_accuracy_pct", "hit_rate_pct",
                "avg_mae_pct", "avg_snr", "avg_pred_move_pct",
                "avg_act_move_pct", "dir_pval", "score"]
    return survivors[out_cols]


def merge(asof_date: date, shard_dir: Path) -> dict:
    parts = [pd.read_csv(p) for p in sorted(shard_dir.glob("shard_*.csv"))]
    failed = []
    for p in sorted(shard_dir.glob("shard_*.csv.failed")):
        failed += [x for x in p.read_text().split("\n") if x]
    df_out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    df_out = df_out.sort_values("dir_accuracy_pct", ascending=False)
    od = out_dir(asof_date)
    od.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(od / "lstm_accuracy_results.csv", index=False)

    surv = rank_candidates(df_out) if not df_out.empty else pd.DataFrame()
    surv.to_csv(od / "lstm_candidates.csv", index=False)
    capped = surv.sort_values("score", ascending=False).head(config.MAX_CANDIDATES) \
        if not surv.empty else surv
    tickers = [str(t) for t in capped.get("ticker", [])]
    summary = {
        "asof": asof_date.isoformat(),
        "members": len(asof.members_live_format(asof_date)),
        "swept": int(len(df_out)),
        "failed": failed,
        "cleared_gates": int(len(surv)),
        "candidates": tickers,
        "mean_dir_accuracy": round(float(df_out["dir_accuracy_pct"].mean()), 2) if len(df_out) else None,
    }
    (od / "stage1.json").write_text(json.dumps(summary, indent=2))
    print(f"[stage1] {asof_date}: swept {summary['swept']}, cleared "
          f"{summary['cleared_gates']}, handed {len(tickers)} downstream", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shard", "merge"])
    ap.add_argument("--asof", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    a = ap.parse_args()
    d = date.fromisoformat(a.asof)
    shard_dir = Path(a.shard_dir) if a.shard_dir else config.DATA_DIR / "stage1" / a.asof
    if a.cmd == "shard":
        run_shard(d, a.shard, a.nshards, shard_dir / f"shard_{a.shard:02d}.csv", a.limit)
    else:
        merge(d, shard_dir)


if __name__ == "__main__":
    main()
