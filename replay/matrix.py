"""Job matrices for GitHub Actions (printed as JSON for `fromJSON`)."""

from __future__ import annotations

import argparse
import json
import math

from . import asof


def stage1(quarter: str, nshards: int) -> list[dict]:
    dates = asof.weekly_asof_dates()
    if quarter != "all":
        q = int(quarter)
        dates = [d for d in dates if (d.month - 1) // 3 + 1 == q]
    return [{"asof": d.isoformat(), "shard": s, "nshards": nshards}
            for d in dates for s in range(nshards)]


def forecasts(fits_per_shard: int) -> list[dict]:
    from .forecasts import jobs
    n = max(1, math.ceil(len(jobs()) / fits_per_shard))
    return [{"shard": s, "nshards": n} for s in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["stage1", "forecasts"])
    ap.add_argument("--quarter", default="1")
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--fits-per-shard", type=int, default=200)
    a = ap.parse_args()
    m = stage1(a.quarter, a.nshards) if a.kind == "stage1" else forecasts(a.fits_per_shard)
    if len(m) > 256:
        raise SystemExit(f"matrix has {len(m)} jobs; GitHub allows 256 per workflow run")
    print(json.dumps({"include": m}))


if __name__ == "__main__":
    main()
