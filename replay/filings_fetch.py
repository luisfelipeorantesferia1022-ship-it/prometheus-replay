"""
Download S&P 500 10-K / 10-Q filings from EDGAR and turn them into the
word-similarity signals (F1, F2) and the paragraph diffs the LLM reads (F3).
Runs on GitHub Actions (sec.gov is not reachable from the PC or the Claude
workspace). Pre-registration: results/filings/PREREGISTRATION.md.

    python -m replay.filings_fetch --shard 0 --nshards 20 --out data/filings
    python -m replay.filings_fetch --ciks 0000320193,0000310158 --out data/filings --samples
    python -m replay.filings_fetch --merge --out data/filings

Needs SEC_USER_AGENT ("Name email"), which EDGAR requires. Requests are spaced
so the whole job stays under EDGAR's 10 requests a second (SEC_MIN_INTERVAL
seconds between requests in each shard).

Per shard, written to <out>/shard_<k>/:
  index.csv            one row per filing: dates, form, section lengths, prior,
                       F1/F2 cosine similarities, diff counts
  payloads.jsonl.gz    unmasked paragraph diffs per section (truncated at 1.5x
                       the pre-registered caps; masking happens when requests
                       are built)
  cikmeta.json         EDGAR names, former names and tickers per CIK
  errors.csv           what could not be downloaded or parsed
No market data is touched here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from . import filings_text as ft

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "results" / "filings" / "universe_ciks.csv"
FIRST_FILED = "2004-01-01"
LAST_FILED = "2026-10-23"
FORMS = {"10-K", "10-Q"}
PRIOR_MIN_DAYS, PRIOR_MAX_DAYS = 330, 400
CAPS = {"rf": 12000, "legal": 6000, "mdna": 12000}       # pre-registered (chars)
STORE_FACTOR = 1.5

INDEX_COLS = ["cik", "accession", "form", "filing_date", "accepted", "report_date",
              "primary_doc", "doc_bytes", "text_chars", "rf_chars", "legal_chars",
              "mdna_chars", "prior_accession", "prior_report_date", "f1_cos", "f2_cos",
              "rf_added", "rf_modified", "rf_removed", "legal_added", "legal_modified",
              "legal_removed", "mdna_added", "mdna_modified", "mdna_removed",
              "rf_status", "legal_status", "mdna_status"]


class Edgar:
    def __init__(self, ua: str, min_interval: float):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last = 0.0
        self.n = 0

    def get(self, url: str, tries: int = 5) -> requests.Response | None:
        for k in range(tries):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n += 1
            try:
                r = self.s.get(url, timeout=120)
            except requests.RequestException as e:
                print(f"[edgar] {type(e).__name__} {url} (try {k + 1})", flush=True)
                time.sleep(5 * (k + 1))
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            print(f"[edgar] HTTP {r.status_code} {url} (try {k + 1})", flush=True)
            time.sleep(10 * (k + 1) if r.status_code in (403, 429, 503) else 3)
        return None


def list_filings(ed: Edgar, cik: str) -> tuple[list[dict], dict]:
    r = ed.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if r is None:
        raise RuntimeError("submissions json not found")
    sub = r.json()
    meta = {"name": sub.get("name"), "tickers": sub.get("tickers") or [],
            "formerNames": [f.get("name") for f in sub.get("formerNames") or []]}
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files") or []:
        if (f.get("filingTo") or "9999") < FIRST_FILED:
            continue
        r2 = ed.get(f"https://data.sec.gov/submissions/{f['name']}")
        if r2 is not None:
            blocks.append(r2.json())
    rows = []
    for b in blocks:
        n = len(b.get("accessionNumber", []))
        for i in range(n):
            form = b["form"][i]
            if form not in FORMS:
                continue
            fd = b["filingDate"][i]
            if not (FIRST_FILED <= fd <= LAST_FILED):
                continue
            rows.append({"cik": cik, "accession": b["accessionNumber"][i], "form": form,
                         "filing_date": fd, "accepted": b["acceptanceDateTime"][i],
                         "report_date": b["reportDate"][i],
                         "primary_doc": b["primaryDocument"][i]})
    rows = {r["accession"]: r for r in rows}           # files pages can overlap recent
    return sorted(rows.values(), key=lambda r: r["filing_date"]), meta


def _main_document(raw: str, form: str) -> str:
    """A full SGML submission (.txt) holds several <DOCUMENT>s; keep the 10-K/10-Q."""
    if "<DOCUMENT>" not in raw[:5000].upper() and "<DOCUMENT>" not in raw.upper()[:200000]:
        return raw
    for doc in re.split(r"(?i)<DOCUMENT>", raw)[1:]:
        m = re.search(r"(?i)<TYPE>\s*([^\s<]+)", doc)
        if m and m.group(1).upper().startswith(form):
            t = re.search(r"(?is)<TEXT>(.*?)(?:</TEXT>|$)", doc)
            return t.group(1) if t else doc
    return raw


def fetch_doc(ed: Edgar, row: dict) -> tuple[str, int]:
    cik_int = int(row["cik"])
    acc = row["accession"].replace("-", "")
    doc = row["primary_doc"] or ""
    if doc:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"
    else:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{row['accession']}.txt"
    r = ed.get(url)
    if r is None:
        raise RuntimeError(f"document not found {url}")
    try:
        raw = r.content.decode("utf-8")
    except UnicodeDecodeError:
        raw = r.content.decode("cp1252", errors="replace")
    raw = _main_document(raw, row["form"])
    return raw, len(r.content)


def _store_diff(d: dict, cap: int) -> dict:
    """Keep the paragraphs in the pre-registered order (added, modified, removed)
    up to STORE_FACTOR x cap characters, plus totals for the 'not shown' note."""
    limit = cap * STORE_FACTOR
    used = 0
    keep = {"added": [], "modified": [], "removed": []}
    for kind in ("added", "modified", "removed"):
        for item in d[kind]:
            n = len(item) if isinstance(item, str) else len(item[0]) + len(item[1])
            if used + n > limit:
                break
            keep[kind].append(item)
            used += n
    tot = {k: len(d[k]) for k in ("added", "modified", "removed")}
    chars = sum(len(x) for x in d["added"]) + sum(len(x) for x in d["removed"]) + \
        sum(len(a) + len(b) for a, b in d["modified"])
    return {**keep, "total": tot, "total_chars": chars}


def process_cik(ed: Edgar, cik: str, samples: list | None = None):
    filings, meta = list_filings(ed, cik)
    parsed, errors = [], []
    for row in filings:
        try:
            raw, nbytes = fetch_doc(ed, row)
            text = ft.html_to_text(raw)
            secs = ft.find_sections(text, row["form"])
            parsed.append({**row, "doc_bytes": nbytes, "text_chars": len(text),
                           "wc": ft.word_counts(text),
                           "wc_rf": ft.word_counts(secs["rf"]) if secs["rf"] else None,
                           "secs": secs})
            if samples is not None:
                samples.append({"cik": cik, "accession": row["accession"], "form": row["form"],
                                "filing_date": row["filing_date"],
                                **{f"{k}_head": v[:400] for k, v in secs.items()},
                                **{f"{k}_tail": v[-300:] for k, v in secs.items()},
                                "text_head": text[:300]})
        except Exception as e:                               # noqa: BLE001
            errors.append({"cik": cik, "accession": row["accession"], "error": f"{type(e).__name__}: {e}"[:300]})
    by_form: dict[str, list] = {}
    for p in parsed:
        by_form.setdefault(p["form"], []).append(p)
    index, payloads = [], []
    for p in parsed:
        out = {c: p.get(c) for c in ("cik", "accession", "form", "filing_date", "accepted",
                                     "report_date", "primary_doc", "doc_bytes", "text_chars")}
        for k in ("rf", "legal", "mdna"):
            out[f"{k}_chars"] = len(p["secs"][k])
        prior = None
        try:
            rd = date.fromisoformat(p["report_date"])
            best = None
            for q in by_form[p["form"]]:
                if not q["report_date"] or q is p:
                    continue
                gap = (rd - date.fromisoformat(q["report_date"])).days
                if PRIOR_MIN_DAYS <= gap <= PRIOR_MAX_DAYS and (best is None or abs(gap - 365) < best[0]):
                    best = (abs(gap - 365), q)
            prior = best[1] if best else None
        except (TypeError, ValueError):
            prior = None
        out["prior_accession"] = prior["accession"] if prior else ""
        out["prior_report_date"] = prior["report_date"] if prior else ""
        out["f1_cos"] = ft.cosine(p["wc"], prior["wc"]) if prior else float("nan")
        out["f2_cos"] = (ft.cosine(p["wc_rf"], prior["wc_rf"])
                         if prior and p["wc_rf"] and prior["wc_rf"] else float("nan"))
        rec = {"cik": cik, "accession": p["accession"], "form": p["form"],
               "prior_accession": out["prior_accession"], "sections": {}}
        for k in ("rf", "legal", "mdna"):
            if not prior or not p["secs"][k] or not prior["secs"][k]:
                status, d = "absent", None
            else:
                d = ft.diff_paragraphs(p["secs"][k], prior["secs"][k])
                status = "changed" if (d["added"] or d["modified"] or d["removed"]) else "nochange"
            out[f"{k}_status"] = status
            for kind in ("added", "modified", "removed"):
                out[f"{k}_{kind}"] = len(d[kind]) if d else ""
            rec["sections"][k] = {"status": status, **(_store_diff(d, CAPS[k]) if d and status == "changed" else {})}
        index.append(out)
        if prior:
            payloads.append(rec)
    return index, payloads, meta, errors


def run_shard(ciks: list[str], out: Path, samples: bool) -> None:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        sys.exit("SEC_USER_AGENT is not set (EDGAR needs 'Name email')")
    ed = Edgar(ua, float(os.environ.get("SEC_MIN_INTERVAL", "2.2")))
    out.mkdir(parents=True, exist_ok=True)
    smp = [] if samples else None
    all_meta, n_err = {}, 0
    t0 = time.time()
    with open(out / "index.csv", "w", newline="") as fi, \
            gzip.open(out / "payloads.jsonl.gz", "wt", encoding="utf-8") as fp, \
            open(out / "errors.csv", "w", newline="") as fe:
        wi = csv.DictWriter(fi, fieldnames=INDEX_COLS)
        wi.writeheader()
        we = csv.DictWriter(fe, fieldnames=["cik", "accession", "error"])
        we.writeheader()
        for j, cik in enumerate(ciks):
            try:
                index, payloads, meta, errors = process_cik(ed, cik, smp)
            except Exception as e:                           # noqa: BLE001
                we.writerow({"cik": cik, "accession": "", "error": f"{type(e).__name__}: {e}"[:300]})
                n_err += 1
                continue
            all_meta[cik] = meta
            for r in index:
                wi.writerow(r)
            for p in payloads:
                fp.write(json.dumps(p) + "\n")
            for e in errors:
                we.writerow(e)
            n_err += len(errors)
            print(f"[filings] {j + 1}/{len(ciks)} CIK {cik} {meta['name']}: {len(index)} filings, "
                  f"{len(errors)} errors, {ed.n} requests, {time.time() - t0:.0f}s", flush=True)
    (out / "cikmeta.json").write_text(json.dumps(all_meta, indent=1))
    if smp is not None:
        (out / "samples.json").write_text(json.dumps(smp, indent=1))
    print(f"[filings] done: {len(ciks)} CIKs, {n_err} errors, {ed.n} requests", flush=True)


def merge(out: Path, results: Path) -> None:
    """Combine shards; write the committed index (SEC-derived numbers only) and
    parse statistics. Payloads stay in the Actions artifact."""
    shards = sorted(out.glob("shard_*"))
    idx = pd.concat([pd.read_csv(s / "index.csv", dtype={"cik": str}) for s in shards
                     if (s / "index.csv").exists()], ignore_index=True)
    err = pd.concat([pd.read_csv(s / "errors.csv", dtype={"cik": str}) for s in shards
                     if (s / "errors.csv").exists()], ignore_index=True)
    meta = {}
    for s in shards:
        if (s / "cikmeta.json").exists():
            meta.update(json.loads((s / "cikmeta.json").read_text()))
    with gzip.open(out / "payloads_all.jsonl.gz", "wt", encoding="utf-8") as fo:
        for s in shards:
            p = s / "payloads.jsonl.gz"
            if p.exists():
                with gzip.open(p, "rt", encoding="utf-8") as fi:
                    for line in fi:
                        fo.write(line)
    (out / "cikmeta_all.json").write_text(json.dumps(meta))
    results.mkdir(parents=True, exist_ok=True)
    idx.sort_values(["cik", "filing_date", "accession"]).to_csv(
        results / "filings_index.csv.gz", index=False)
    err.to_csv(results / "fetch_errors.csv", index=False)
    (results / "cikmeta.json").write_text(json.dumps(meta, indent=0))
    stats = parse_stats(idx, err)
    (results / "parse_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def parse_stats(idx: pd.DataFrame, err: pd.DataFrame) -> dict:
    idx = idx.copy()
    idx["year"] = idx["filing_date"].str[:4]
    s = {"filings": int(len(idx)), "ciks": int(idx["cik"].nunique()),
         "errors": int(len(err)), "by_form": idx["form"].value_counts().to_dict(),
         "with_prior": float((idx["prior_accession"].fillna("") != "").mean())}
    for k in ("rf", "legal", "mdna"):
        s[f"{k}_found"] = {f: float((g[f"{k}_chars"] > 0).mean()) for f, g in idx.groupby("form")}
        s[f"{k}_status"] = idx[f"{k}_status"].value_counts().to_dict()
    s["rf_found_by_year_10K"] = {y: round(float((g["rf_chars"] > 0).mean()), 3)
                                 for y, g in idx[idx.form == "10-K"].groupby("year")}
    s["mdna_found_by_year"] = {y: round(float((g["mdna_chars"] > 0).mean()), 3)
                               for y, g in idx.groupby("year")}
    s["f1_quantiles"] = idx["f1_cos"].quantile([.01, .1, .5, .9, .99]).round(4).to_dict()
    s["f2_quantiles"] = idx["f2_cos"].quantile([.01, .1, .5, .9, .99]).round(4).to_dict()
    hrs = pd.to_datetime(idx["accepted"], errors="coerce", utc=True).dt.hour
    s["accepted_hour_counts"] = hrs.value_counts().sort_index().to_dict()
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--ciks", default="")
    ap.add_argument("--out", default="data/filings")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--results", default="results/filings")
    a = ap.parse_args()
    out = Path(a.out)
    if a.merge:
        merge(out, Path(a.results))
        return
    if a.ciks:
        ciks = [c.strip().zfill(10) for c in a.ciks.split(",") if c.strip()]
    else:
        u = pd.read_csv(UNIVERSE, dtype={"cik": str})
        allc = sorted(u["cik"].str.zfill(10).unique())
        ciks = allc[a.shard::a.nshards]
    run_shard(ciks, out / f"shard_{a.shard:02d}", a.samples)


if __name__ == "__main__":
    main()
