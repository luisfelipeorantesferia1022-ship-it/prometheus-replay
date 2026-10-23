"""
Text processing for the SEC filings test (results/filings/PREREGISTRATION.md).

    html_to_text(raw, is_html)      primary document -> plain text, one block per line
    find_sections(text, form)       {"rf", "legal", "mdna"} -> section text or ""
    word_counts(text)               lowercase [a-z]+ token counts
    cosine(a, b)                    cosine similarity of two Counters
    paragraphs(text)                lines with at least 8 words
    diff_paragraphs(cur, prior)     added / modified / removed paragraphs
    mask(text, names, tickers)      hide company, tickers, years, dates, people

No network, no market data. Pure functions so tests can run them on fixtures.
"""

from __future__ import annotations

import html as _html
import math
import re
from collections import Counter

import numpy as np

MIN_SECTION_CHARS = 500
MIN_PARA_WORDS = 8
J_UNCHANGED = 0.9
J_MODIFIED = 0.5

# ─────────────────────────────────────────────────────────────────────────────
#  HTML -> text
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
          "table", "ul", "ol", "center", "blockquote", "pre", "hr", "section",
          "article", "header", "footer", "title"}
_CELL = {"td", "th"}
_DROP = {"script", "style", "head", "ix:header", "noscript"}


def _clean_lines(text: str) -> str:
    text = (text.replace("\xa0", " ").replace("​", "").replace(" ", " ")
            .replace(" ", " ").replace(" ", " ").replace("\r", "\n")
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"'))
    out = []
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t\f\v]+", " ", ln).strip()
        out.append(ln)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _plain_text_blocks(text: str) -> str:
    """Hard-wrapped .txt filings: join lines within blank-line separated blocks,
    so each paragraph becomes one line (as in the HTML path)."""
    text = re.sub(r"<[^>]{1,200}>", " ", text)          # stray SGML tags
    text = _html.unescape(text)
    text = text.replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n", text)
    lines = []
    for b in blocks:
        parts = [re.sub(r"\s+", " ", ln).strip() for ln in b.split("\n")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        # Keep short heading-like lines (e.g. "ITEM 1A. RISK FACTORS") on their own
        # line so section regexes anchored at line start still find them.
        cur = []
        for p in parts:
            if re.match(r"(?i)^(part\s+i{1,2}\b|item\s*\d)", p) and len(p) < 120:
                if cur:
                    lines.append(" ".join(cur)); cur = []
                lines.append(p)
            else:
                cur.append(p)
        if cur:
            lines.append(" ".join(cur))
    return _clean_lines("\n".join(lines))


def html_to_text(raw: bytes | str, is_html: bool | None = None) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if is_html is None:
        is_html = bool(re.search(r"(?i)<(html|body|p|div|table|font)\b", raw[:200000]))
    if not is_html:
        return _plain_text_blocks(raw)
    import lxml.html
    from lxml import etree
    # Old EDGAR HTML is often wrapped in <DOCUMENT><TEXT> SGML; lxml copes.
    raw = re.sub(r"(?is)<\?xml[^>]*\?>", "", raw)
    try:
        root = lxml.html.document_fromstring(raw)
    except (etree.ParserError, ValueError):
        return _plain_text_blocks(raw)
    drop = []
    for el in root.iter():
        if not isinstance(el.tag, str):          # comments, processing instructions
            drop.append(el)
            continue
        style = (el.get("style") or "").replace(" ", "").lower()
        if el.tag.lower() in _DROP or "display:none" in style:
            drop.append(el)
    for el in drop:
        if el.getparent() is not None:
            el.drop_tree()                       # keeps the element's tail text
    for el in root.iter():
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        if tag in _BLOCK:
            el.text = "\n" + (el.text or "")
            el.tail = "\n" + (el.tail or "")
        elif tag in _CELL:
            el.tail = " " + (el.tail or "")
    return _clean_lines(root.text_content())


# ─────────────────────────────────────────────────────────────────────────────
#  Sections
# ─────────────────────────────────────────────────────────────────────────────

_SEP = r"[\s\.\:\-–—‒―,]*"
_ITEM = r"^[ \t]*(?:part[ \t]+i{1,2}[ \t]*" + _SEP + r")?item[ \t]*"

_PATTERNS = {
    "10-K": {
        "rf": (_ITEM + r"1a\b" + _SEP + r"risk\s+factors", _ITEM + r"(?:1b|2)\b"),
        "legal": (_ITEM + r"3\b" + _SEP + r"legal\s+proceedings", _ITEM + r"(?:4|4a)\b"),
        "mdna": (_ITEM + r"7\b" + _SEP + r"management", _ITEM + r"(?:7a|8)\b"),
    },
    "10-Q": {
        "rf": (_ITEM + r"1a\b" + _SEP + r"risk\s+factors", _ITEM + r"(?:2|3|4|5|6)\b"),
        "legal": (_ITEM + r"1\b" + _SEP + r"legal\s+proceedings", _ITEM + r"(?:1a|2)\b"),
        "mdna": (_ITEM + r"2\b" + _SEP + r"management", _ITEM + r"(?:3|4)\b"),
    },
}
_COMPILED = {f: {k: (re.compile(s, re.I | re.M), re.compile(e, re.I | re.M))
                 for k, (s, e) in d.items()} for f, d in _PATTERNS.items()}


def find_sections(text: str, form: str) -> dict[str, str]:
    """For each section: among all heading matches, the one whose span to the
    next end heading is longest (skips the table of contents). A start with no
    end heading after it is ignored. Sections under 500 characters -> ""."""
    pats = _COMPILED["10-K" if form.upper().startswith("10-K") else "10-Q"]
    out = {}
    for key, (start_re, end_re) in pats.items():
        best = ""
        starts = list(start_re.finditer(text))
        for i, m in enumerate(starts):
            e = end_re.search(text, m.end())
            if not e:
                continue
            # A later heading of the same item before this end heading wins
            # (a table-of-contents line whose end heading is missing).
            if i + 1 < len(starts) and starts[i + 1].start() < e.start():
                continue
            body = text[m.start():e.start()].strip()
            if len(body) > len(best):
                best = body
        out[key] = best if len(best) >= MIN_SECTION_CHARS else ""
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity
# ─────────────────────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z]+")


def word_counts(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
#  Paragraph diff
# ─────────────────────────────────────────────────────────────────────────────

_PWORD = re.compile(r"[a-z#]+")


def paragraphs(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if len(ln.split()) >= MIN_PARA_WORDS]


def _wordset(p: str) -> frozenset:
    return frozenset(_PWORD.findall(re.sub(r"\d", "#", p.lower())))


def _jaccard_matrix(cur: list[frozenset], pri: list[frozenset]) -> np.ndarray:
    vocab: dict[str, int] = {}
    def rows(sets):
        from scipy.sparse import csr_matrix
        ind, ptr = [], [0]
        for s in sets:
            ind.extend(vocab.setdefault(w, len(vocab)) for w in s)
            ptr.append(len(ind))
        return ind, ptr
    from scipy.sparse import csr_matrix
    ci, cp = rows(cur)
    pi, pp = rows(pri)
    n = len(vocab)
    A = csr_matrix((np.ones(len(ci), dtype=np.float32), ci, cp), shape=(len(cur), n))
    B = csr_matrix((np.ones(len(pi), dtype=np.float32), pi, pp), shape=(len(pri), n))
    inter = (A @ B.T).toarray()
    la = np.array([len(s) for s in cur], dtype=np.float32)[:, None]
    lb = np.array([len(s) for s in pri], dtype=np.float32)[None, :]
    union = la + lb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    return J


def diff_paragraphs(cur_text: str, prior_text: str) -> dict:
    """Pre-registered matching: each current paragraph's best prior match by
    Jaccard of digit-normalized word sets. >= 0.9 unchanged, 0.5-0.9 modified,
    < 0.5 added. Prior paragraphs with no current match >= 0.5 are removed."""
    cur, pri = paragraphs(cur_text), paragraphs(prior_text)
    out = {"added": [], "modified": [], "removed": [], "n_cur": len(cur), "n_prior": len(pri)}
    if not cur and not pri:
        return out
    if not pri:
        out["added"] = cur
        return out
    if not cur:
        out["removed"] = pri
        return out
    J = _jaccard_matrix([_wordset(p) for p in cur], [_wordset(p) for p in pri])
    best_j = J.max(axis=1)
    best_i = J.argmax(axis=1)
    for k, p in enumerate(cur):
        if best_j[k] >= J_UNCHANGED:
            continue
        if best_j[k] >= J_MODIFIED:
            out["modified"].append([pri[best_i[k]], p])
        else:
            out["added"].append(p)
    covered = J.max(axis=0) >= J_MODIFIED
    out["removed"] = [p for k, p in enumerate(pri) if not covered[k]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Masking
# ─────────────────────────────────────────────────────────────────────────────

_SUFFIX = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
           "limited", "plc", "llc", "lp", "l", "p", "holdings", "holding", "group",
           "the", "de", "nv", "n", "v", "sa", "ag", "trust", "bancorp", "cos", "companies"}
_MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|October|"
           r"November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|"
           r"Oct\.?|Nov\.?|Dec\.?)")


def name_variants(names: list[str]) -> list[str]:
    """Full names and names without legal suffixes (EDGAR style, e.g.
    'APPLE INC.', 'MERCK & CO INC /NJ/'). Variants shorter than 4 letters are
    dropped (too likely to hit ordinary words)."""
    out = set()
    for n in names:
        if not n:
            continue
        n = re.sub(r"/[A-Z]{2,3}/?", " ", n)          # /DE/, /NJ/
        n = re.sub(r"\s+", " ", n).strip(" ,.")
        toks = re.findall(r"[A-Za-z0-9&']+", n)
        if not toks:
            continue
        out.add(" ".join(toks))
        while toks and toks[-1].lower().strip(".") in _SUFFIX:
            toks = toks[:-1]
        while toks and toks[0].lower() == "the":
            toks = toks[1:]
        if toks:
            out.add(" ".join(toks))
    return sorted((v for v in out if len(re.sub(r"[^A-Za-z]", "", v)) >= 4),
                  key=len, reverse=True)


def _name_regex(v: str) -> str:
    toks = re.findall(r"[A-Za-z0-9&']+", v)
    return r"\b" + r"[\s,\.\-]*".join(re.escape(t) for t in toks) + r"\b"


def mask(text: str, names: list[str], tickers: list[str]) -> str:
    t = text
    t = re.sub(r"\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b", "[DATE]", t)
    t = re.sub(r"\b(?:19[5-9]\d|20[0-3]\d)\b", "[YEAR]", t)
    t = re.sub(r"\bFY\s?'?\d{2}\b", "FY[YEAR]", t, flags=re.I)
    t = re.sub(r"\b" + _MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:\[YEAR\])?", "[DATE]", t)
    t = re.sub(r"\b(?:Mr|Ms|Mrs|Dr)\.?\s+[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?", "[PERSON]", t)
    for v in name_variants(names):
        t = re.sub(_name_regex(v), "[COMPANY]", t, flags=re.I)
    for tk in sorted({x for x in tickers if x and len(x) >= 2}, key=len, reverse=True):
        t = re.sub(r"\b" + re.escape(tk) + r"\b", "[TICKER]", t)
    return t
