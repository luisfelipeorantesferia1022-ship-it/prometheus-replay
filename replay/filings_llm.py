"""
LLM reading of filing changes (signal F3) for results/filings/PREREGISTRATION.md.
The prompt text below is frozen by the pre-registration commit.

    python -m replay.filings_llm submit --kind pilot|main|l1|l2 --payloads data/filings/payloads_all.jsonl.gz
    python -m replay.filings_llm collect [--wait-hours 5]
    python -m replay.filings_llm retry  --kind main      # resubmit replies that did not parse
    python -m replay.filings_llm status

Kinds:
  pilot  300 random search-period filings (plumbing and cost only, never returns)
  main   every filing with a prior and at least one changed section
  l1     300 random holdout filings: can the model name the masked company?
  l2     600 random holdout filings scored again WITHOUT masking

Budget: results/filings/llm/ledger.json keeps the spend from the token counts
the API reports, priced at list (no batch discount), plus the estimate of every
batch still running. A submit that could push the total over BUDGET_USD is
refused. Needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import filings_text as ft

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "filings"
LLM_DIR = RES / "llm"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 400
TEMPERATURE = 0.0
BUDGET_USD = 250.0
# List prices per million tokens (Claude Haiku 5.5, prompts under 100K tokens),
# used without the 50% batch discount so the guard errs high.
PRICE_IN, PRICE_OUT = 0.10, 0.50
CHARS_PER_TOKEN_EST = 3.0             # conservative (real English is ~4)
BATCH_MAX_REQUESTS = 4000
CAPS = {"rf": 12000, "legal": 6000, "mdna": 12000}
SECTION_TITLE = {"rf": "RISK FACTORS", "legal": "LEGAL PROCEEDINGS",
                 "mdna": "MANAGEMENT'S DISCUSSION AND ANALYSIS"}
SEARCH_FILED = ("2006-01-01", "2018-12-31")
HOLDOUT_FILED = ("2019-01-01", "2026-08-31")
SEED = 20261023

# ─────────────────────────────────────────────────────────────────────────────
#  Frozen prompt
# ─────────────────────────────────────────────────────────────────────────────

_P1 = ("You are a careful equity analyst. You read what changed between two versions "
       "of the same U.S. public company's periodic SEC report: the current report and "
       "the report for the same fiscal period one year earlier.")
_P2_MASKED = ("Company names, ticker symbols, people's names, years and dates have been "
              "masked. Judge only the text you are given. Do not try to identify the "
              "company, and do not use anything you may know about any company, industry "
              "or later events.")
_P2_UNMASKED = ("Judge only the text you are given. Do not use anything you may know about "
                "any company, industry or later events.")
_P3 = ("You will see, for up to three sections (Risk Factors, Legal Proceedings, "
       "Management's Discussion and Analysis), the paragraphs that were ADDED in the "
       "current report, the paragraphs REMOVED since the earlier report, and paragraphs "
       "that were MODIFIED (earlier wording, then current wording). Unchanged paragraphs "
       "are not shown. Long sections may be cut; a note says how much was left out.")
_P4 = ("Decide whether the changes, taken together, reveal that the company's risks or "
       "business prospects got worse or better, from the point of view of a shareholder. "
       "Wording edits that do not change meaning, updated dates, and routine boilerplate "
       "are neutral.")
_P5 = ('Reply with one JSON object and nothing else:\n'
       '{"tone": <integer -3..3>, "materiality": <integer 0..3>, '
       '"new_material_risk": <true|false>, "litigation_escalated": <true|false>, '
       '"litigation_resolved": <true|false>, "liquidity_or_going_concern": <true|false>, '
       '"management_or_auditor_change": <true|false>, '
       '"restructuring_or_impairment": <true|false>, '
       '"government_investigation": <true|false>, "summary": "<at most 30 words>"}\n\n'
       "tone: -3 major adverse new information; -2 clearly adverse; -1 slightly adverse; "
       "0 neutral, mixed or cosmetic; +1 slightly favourable; +2 clearly favourable; "
       "+3 major favourable.\n"
       "materiality: 0 cosmetic or boilerplate only; 1 minor; 2 meaningful; 3 major.\n"
       "The flags are true only if the changes show that event or condition newly "
       "appearing or getting worse (for litigation_resolved: a matter settled, dismissed "
       "or closed).")
SYSTEM = "\n\n".join([_P1, _P2_MASKED, _P3, _P4, _P5])
SYSTEM_UNMASKED = "\n\n".join([_P1, _P2_UNMASKED, _P3, _P4, _P5])
SYSTEM_IDENTIFY = (
    "Below are excerpts from a U.S. public company's periodic SEC report, showing what "
    "changed from the report one year earlier. Company names, tickers, people and dates "
    "are masked. Which company filed this report? If you are not reasonably sure, say "
    "unknown. Reply with one JSON object and nothing else: "
    '{"company": "<name or unknown>", "ticker": "<ticker or unknown>"}')
FLAGS = ["new_material_risk", "litigation_escalated", "litigation_resolved",
         "liquidity_or_going_concern", "management_or_auditor_change",
         "restructuring_or_impairment", "government_investigation"]


# ─────────────────────────────────────────────────────────────────────────────
#  Payload -> user message
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_section(sec: dict, cap: int, m) -> str:
    st = sec.get("status")
    if st == "absent":
        return "(Section not found in one or both reports.)"
    if st == "nochange":
        return "(No changes.)"
    parts, used = [], 0
    shown = {"added": 0, "modified": 0, "removed": 0}
    stop = False
    for kind, label in (("added", "ADDED"), ("modified", "MODIFIED"), ("removed", "REMOVED")):
        lines = []
        for item in sec.get(kind, []):
            if kind == "modified":
                old, new = m(item[0]), m(item[1])
                txt = f"- Earlier: {old}\n  Current: {new}"
                n = len(old) + len(new)
            else:
                txt = f"- {m(item)}"
                n = len(txt) - 2
            if used + n > cap:
                stop = True
                break
            lines.append(txt)
            used += n
            shown[kind] += 1
        if lines:
            parts.append(label + ":\n" + "\n".join(lines))
        if stop:
            break
    tot = sec.get("total", shown)
    left = {k: tot[k] - shown[k] for k in shown}
    if any(left.values()):
        parts.append(f"[Not shown: {left['added']} added, {left['modified']} modified, "
                     f"{left['removed']} removed paragraphs.]")
    return "\n".join(parts) if parts else "(No changes.)"


def user_message(rec: dict, names: list[str], tickers: list[str], masked: bool = True) -> str:
    m = (lambda s: ft.mask(s, names, tickers)) if masked else (lambda s: s)
    out = [f"Report type: {rec['form']}"]
    for k in ("rf", "legal", "mdna"):
        out.append(f"=== {SECTION_TITLE[k]} ===\n" + _fmt_section(rec["sections"][k], CAPS[k], m))
    return "\n\n".join(out)


def needs_call(rec: dict) -> str:
    """'call' if any section changed; 'zero' if some section is found in both
    reports and none changed (tone 0 without a call); 'missing' otherwise."""
    st = [rec["sections"][k]["status"] for k in ("rf", "legal", "mdna")]
    if "changed" in st:
        return "call"
    if "nochange" in st:
        return "zero"
    return "missing"


def parse_reply(text: str) -> dict | None:
    try:
        j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        tone, mat = int(j["tone"]), int(j["materiality"])
        if not (-3 <= tone <= 3 and 0 <= mat <= 3):
            return None
        out = {"tone": tone, "materiality": mat,
               "summary": str(j.get("summary", ""))[:400]}
        for f in FLAGS:
            v = j.get(f)
            out[f] = bool(v) if isinstance(v, bool) else str(v).lower() == "true"
        return out
    except (ValueError, KeyError, TypeError):
        return None


def parse_identify(text: str) -> dict | None:
    try:
        j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return {"company": str(j.get("company", "")), "ticker": str(j.get("ticker", ""))}
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Ledger and batches
# ─────────────────────────────────────────────────────────────────────────────

def _ledger() -> dict:
    p = LLM_DIR / "ledger.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"budget_usd": BUDGET_USD, "spent_usd": 0.0, "input_tokens": 0,
            "output_tokens": 0, "batches": []}


def _save_ledger(L: dict) -> None:
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    (LLM_DIR / "ledger.json").write_text(json.dumps(L, indent=1))


def _cost(tin: float, tout: float) -> float:
    return tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT


def committed_usd(L: dict) -> float:
    return L["spent_usd"] + sum(b["est_usd"] for b in L["batches"] if not b.get("collected"))


def load_payloads(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _names_tickers(cikmeta: dict, universe: pd.DataFrame, cik: str) -> tuple[list[str], list[str]]:
    m = cikmeta.get(cik, {})
    u = universe[universe["cik"] == cik]
    names = [m.get("name") or ""] + list(m.get("formerNames") or []) + list(u["name"])
    tickers = list(m.get("tickers") or []) + list(u["ticker"])
    return names, tickers


def select(kind: str, payloads: list[dict], index: pd.DataFrame) -> list[dict]:
    fd = index.set_index("accession")["filing_date"]
    calls = [p for p in payloads if needs_call(p) == "call"]
    rng = np.random.default_rng(SEED)
    def in_range(p, rg):
        d = fd.get(p["accession"], "")
        return rg[0] <= d <= rg[1]
    if kind == "main":
        return calls
    if kind == "pilot":
        pool = [p for p in calls if in_range(p, SEARCH_FILED)]
        return [pool[i] for i in sorted(rng.choice(len(pool), min(300, len(pool)), replace=False))]
    pool = [p for p in calls if in_range(p, HOLDOUT_FILED)]
    rng = np.random.default_rng(SEED + (1 if kind == "l1" else 2))
    n = 300 if kind == "l1" else 600
    return [pool[i] for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))]


def build_requests(kind: str, recs: list[dict], cikmeta: dict, universe: pd.DataFrame) -> list[dict]:
    reqs = []
    for r in recs:
        names, tickers = _names_tickers(cikmeta, universe, r["cik"])
        masked = kind != "l2"
        system = SYSTEM_IDENTIFY if kind == "l1" else (SYSTEM if masked else SYSTEM_UNMASKED)
        user = user_message(r, names, tickers, masked=masked)
        reqs.append({"custom_id": f"{kind}-{r['accession'].replace('-', '')}",
                     "params": {"model": MODEL, "max_tokens": MAX_TOKENS,
                                "temperature": TEMPERATURE, "system": system,
                                "messages": [{"role": "user", "content": user}]}})
    return reqs


def est_usd(reqs: list[dict]) -> float:
    chars = sum(len(q["params"]["system"]) + len(q["params"]["messages"][0]["content"]) for q in reqs)
    return _cost(chars / CHARS_PER_TOKEN_EST, len(reqs) * MAX_TOKENS)


def submit(kind: str, reqs: list[dict], dry: bool = False) -> None:
    import anthropic
    L = _ledger()
    est = est_usd(reqs)
    print(f"[llm] {kind}: {len(reqs)} requests, estimated ${est:.2f} at list price; "
          f"committed so far ${committed_usd(L):.2f} of ${L['budget_usd']:.0f}", flush=True)
    if committed_usd(L) + est > L["budget_usd"]:
        sys.exit(f"[llm] refused: would exceed the ${L['budget_usd']:.0f} budget")
    if dry:
        return
    client = anthropic.Anthropic()
    for i in range(0, len(reqs), BATCH_MAX_REQUESTS):
        chunk = reqs[i:i + BATCH_MAX_REQUESTS]
        b = client.messages.batches.create(requests=chunk)
        L["batches"].append({"id": b.id, "kind": kind, "n": len(chunk), "est_usd": est_usd(chunk),
                             "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "collected": False})
        _save_ledger(L)
        print(f"[llm] submitted {b.id} ({len(chunk)} requests)", flush=True)


def collect(wait_hours: float) -> None:
    import anthropic
    client = anthropic.Anthropic()
    deadline = time.time() + wait_hours * 3600
    while True:
        L = _ledger()
        open_b = [b for b in L["batches"] if not b.get("collected")]
        if not open_b:
            print("[llm] nothing to collect", flush=True)
            return
        for b in open_b:
            info = client.messages.batches.retrieve(b["id"])
            if info.processing_status != "ended":
                continue
            rows, tin, tout = [], 0, 0
            for e in client.messages.batches.results(b["id"]):
                acc = e.custom_id.split("-", 1)[1]
                row = {"accession": acc, "kind": b["kind"], "result": e.result.type}
                if e.result.type == "succeeded":
                    msg = e.result.message
                    text = "".join(getattr(c, "text", "") for c in msg.content)
                    tin += msg.usage.input_tokens
                    tout += msg.usage.output_tokens
                    row.update({"input_tokens": msg.usage.input_tokens,
                                "output_tokens": msg.usage.output_tokens, "raw": text[:2000]})
                    parsed = (parse_identify(text) if b["kind"].startswith("l1")
                              else parse_reply(text))
                    row["parse_ok"] = parsed is not None
                    row.update(parsed or {})
                else:
                    row["parse_ok"] = False
                rows.append(row)
            out = LLM_DIR / f"replies_{b['kind']}.csv.gz"
            df = pd.DataFrame(rows)
            if out.exists():
                df = pd.concat([pd.read_csv(out, dtype={"accession": str}), df], ignore_index=True)
            df.to_csv(out, index=False)
            b["collected"] = True
            b["input_tokens"], b["output_tokens"] = tin, tout
            b["usd_list"] = round(_cost(tin, tout), 4)
            L["input_tokens"] += tin
            L["output_tokens"] += tout
            L["spent_usd"] = round(L["spent_usd"] + _cost(tin, tout), 4)
            _save_ledger(L)
            print(f"[llm] collected {b['id']} {b['kind']}: {len(rows)} replies, "
                  f"{tin:,} in / {tout:,} out tokens, ${_cost(tin, tout):.2f} at list", flush=True)
        if time.time() > deadline:
            print("[llm] stopping; batches still running", flush=True)
            return
        if any(not b.get("collected") for b in _ledger()["batches"]):
            time.sleep(60)


def failed_accessions(kind: str) -> set[str]:
    p = LLM_DIR / f"replies_{kind}.csv.gz"
    if not p.exists():
        return set()
    df = pd.read_csv(p, dtype={"accession": str})
    ok = set(df.loc[df["parse_ok"] == True, "accession"])        # noqa: E712
    return set(df["accession"]) - ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "collect", "retry", "status", "preview"])
    ap.add_argument("--kind", default="pilot")
    ap.add_argument("--payloads", default="data/filings/payloads_all.jsonl.gz")
    ap.add_argument("--cikmeta", default="data/filings/cikmeta_all.json")
    ap.add_argument("--wait-hours", type=float, default=5.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        L = _ledger()
        print(json.dumps({k: v for k, v in L.items() if k != "batches"}, indent=1))
        for b in L["batches"]:
            print(b)
        return
    if a.cmd == "collect":
        collect(a.wait_hours)
        return
    payloads = load_payloads(Path(a.payloads))
    index = pd.read_csv(RES / "filings_index.csv.gz", dtype={"cik": str})
    cikmeta = json.loads(Path(a.cikmeta).read_text())
    universe = pd.read_csv(RES / "universe_ciks.csv", dtype={"cik": str})
    if a.cmd == "retry":
        bad = failed_accessions(a.kind)
        recs = [p for p in payloads if p["accession"].replace("-", "") in bad]
        reqs = build_requests(a.kind, recs, cikmeta, universe)
        for q in reqs:
            q["custom_id"] = q["custom_id"].replace(f"{a.kind}-", f"{a.kind}r-", 1)
        submit(a.kind + "r", reqs, a.dry)
        return
    recs = select(a.kind, payloads, index)
    reqs = build_requests(a.kind, recs, cikmeta, universe)
    if a.cmd == "preview":
        LLM_DIR.mkdir(parents=True, exist_ok=True)
        lens = [len(q["params"]["messages"][0]["content"]) for q in reqs]
        prev = {"kind": a.kind, "n": len(reqs), "est_usd_list": round(est_usd(reqs), 2),
                "user_chars_quantiles": pd.Series(lens).quantile([.1, .5, .9, .99]).to_dict() if lens else {},
                "examples": [{"custom_id": q["custom_id"],
                              "user": q["params"]["messages"][0]["content"][:6000]} for q in reqs[:5]]}
        (LLM_DIR / f"preview_{a.kind}.json").write_text(json.dumps(prev, indent=1))
        print(json.dumps({k: v for k, v in prev.items() if k != "examples"}, indent=1))
        return
    submit(a.kind, reqs, a.dry)


if __name__ == "__main__":
    main()
