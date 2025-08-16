"""
Lightweight post-filtering for vector results.

Goal: remove obvious non-content chunks (references, affiliations, long author lists,
boilerplate citations) while preserving scientific paragraphs.
"""

from __future__ import annotations

import re
from typing import List
from langchain.schema import Document


# Heuristics (can be tuned)
REF_HEADERS = re.compile(r"^(references|acknowledg(e)?ments|bibliography)\b", re.IGNORECASE)
AFFILIATION_HINTS = re.compile(r"(affiliation|department|university|institute|laboratory)\b", re.IGNORECASE)
LONG_AUTHOR_LINE = re.compile(r"^(?:[A-Z][a-z]+\s[A-Z][a-z]+(?:,|\sand\s))+[A-Z][a-z]+\s[A-Z][a-z]+\.?$")
MANY_EMAILS_URLS = re.compile(r"(mailto:|@|doi:|arxiv\.org|https?://)")
LOW_ALPHA_RATIO_MIN = 0.3  # keep chunks with >=30% alphabetic content
MIN_SENTENCES = 1          # require at least one sentence-like period


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text))


def _is_non_content(text: str) -> bool:
    line0 = text.strip().splitlines()[0] if text else ""
    if REF_HEADERS.match(line0):
        return True
    if AFFILIATION_HINTS.search(text) and text.count("@") >= 1:
        return True
    if LONG_AUTHOR_LINE.match(line0):
        return True
    if len(re.findall(MANY_EMAILS_URLS, text)) >= 3:
        return True
    if _alpha_ratio(text) < LOW_ALPHA_RATIO_MIN:
        return True
    if text.count(".") < MIN_SENTENCES:
        return True
    return False


def filter_documents(docs: List[Document]) -> List[Document]:
    """Return a filtered list of docs by dropping obvious non-content chunks."""
    if not docs:
        return docs
    kept: List[Document] = []
    for d in docs:
        txt = getattr(d, "page_content", "") or ""
        if not _is_non_content(txt):
            kept.append(d)
    # If everything is filtered out, fall back to originals to avoid empty answers
    return kept if kept else docs


