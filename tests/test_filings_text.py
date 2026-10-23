"""Synthetic checks for replay/filings_text.py (no network, no market data)."""
from replay import filings_text as ft
from replay import filings_llm as fl
from replay import filings_fetch as ff


def _doc(rf_body: str) -> bytes:
    return (b"<html><body><div style='display:none'><ix:header>2020 hidden</ix:header></div>"
            b"<table><tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr></table>"
            b"<p>ITEM&#160;1A. RISK FACTORS</p>" + rf_body.encode() +
            b"<p>Item 1B. Unresolved Staff Comments</p><p>None.</p></body></html>")


def test_sections_skip_toc_and_hidden():
    body = "<p>" + "Competition may reduce our margins in every market we serve. " * 20 + "</p>"
    t = ft.html_to_text(_doc(body))
    assert "hidden" not in t
    s = ft.find_sections(t, "10-K")
    assert s["rf"].startswith("ITEM 1A. RISK FACTORS") and len(s["rf"]) > 500
    assert s["legal"] == "" and s["mdna"] == ""


def test_diff_ignores_number_changes():
    cur = "Revenue rose 10% to $5 billion in the year on strong demand for products.\nA new lawsuit was filed against us by the government alleging fraud in sales."
    pri = "Revenue rose 7% to $4 billion in the year on strong demand for products.\nWe depend on a few suppliers and losing one would hurt results for some time."
    d = ft.diff_paragraphs(cur, pri)
    assert len(d["added"]) == 1 and "lawsuit" in d["added"][0]
    assert len(d["removed"]) == 1 and d["modified"] == []


def test_mask():
    m = ft.mask("Apple Inc. (AAPL) said on March 3, 2021 that Mr. Tim Cook leaves in 2022.", ["APPLE INC"], ["AAPL"])
    assert "Apple" not in m and "AAPL" not in m and "2021" not in m and "Cook" not in m


def test_payload_rules():
    d = ft.diff_paragraphs("A brand new risk about cyber attacks on our systems could hurt us.", "")
    rec = {"form": "10-Q", "sections": {"rf": {"status": "changed", **ff._store_diff(d, 12000)},
                                        "legal": {"status": "absent"}, "mdna": {"status": "nochange"}}}
    assert fl.needs_call(rec) == "call"
    rec["sections"]["rf"] = {"status": "nochange"}
    assert fl.needs_call(rec) == "zero"
    rec["sections"]["rf"] = {"status": "absent"}; rec["sections"]["mdna"] = {"status": "absent"}
    assert fl.needs_call(rec) == "missing"
    assert fl.parse_reply('{"tone": 4, "materiality": 1}') is None
    assert fl.parse_reply('x {"tone": -1, "materiality": 1} y')["tone"] == -1
