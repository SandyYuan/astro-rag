"""
Corpus-aware reranking utilities to prioritize primary-author sources.

This module provides a deterministic reranker that applies a strong prior
to documents whose metadata.source path begins with a configured prefix
(e.g., "papers/"). It preserves the base ordering within the same group
and enforces a minimum share of primary documents when available.
"""

from __future__ import annotations

import os
import math
from typing import List, Tuple, Optional

from langchain.schema import Document


def _is_primary_source(doc: Document, prefix: str) -> bool:
    source = ""
    try:
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = str(doc.metadata.get("source", "") or "")
    except Exception:
        source = ""
    return bool(prefix) and source.startswith(prefix)


def _extract_year(doc: Document, field: str) -> Optional[int]:
    """Extract an integer publication year from Document.metadata[field].
    No I/O; only use in-memory metadata. Returns None if unavailable/invalid.
    """
    try:
        if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
            return None
        val = doc.metadata.get(field)
        if val is None:
            # common alternative key
            val = doc.metadata.get("published_year")
        if val is None:
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip()
        # Take first 4-digit year in string if present
        if len(s) >= 4:
            for i in range(len(s) - 3):
                chunk = s[i : i + 4]
                if chunk.isdigit():
                    y = int(chunk)
                    if 1800 <= y <= 2200:
                        return y
        return None
    except Exception:
        return None


def rerank_with_primary_weight(
    documents: List[Document],
    primary_prefix: str = "papers/",
    primary_boost: float = 2.0,
    min_primary_share: float = 0.8,
    final_k: int | None = 5,
    recency_enabled: bool = True,
    recency_field: str = "year",
    recency_missing_last: bool = True,
) -> List[Document]:
    """
    Rerank docs with a very strong prior for primary-author sources.

    Strategy:
    - Partition into primary vs non-primary by metadata.source prefix.
    - Preserve original order within each partition (stable).
    - Ensure at least floor(min_primary_share * final_k) primaries if available.
    - Prefer to fill remaining slots with additional primaries first, then non-primary.

    Note: primary_boost is accepted for compatibility but the ordering is primarily
    driven by partition + min_share to reflect a strong corpus prior.
    """
    if not documents:
        return documents

    n = len(documents)
    k = final_k if (final_k is not None and final_k > 0) else n

    indexed_docs: List[Tuple[int, Document]] = list(enumerate(documents))
    primary_list_idx: List[Tuple[int, Document]] = [(i, d) for i, d in indexed_docs if _is_primary_source(d, primary_prefix)]
    non_primary_list_idx: List[Tuple[int, Document]] = [(i, d) for i, d in indexed_docs if not _is_primary_source(d, primary_prefix)]

    def sort_by_recency(items: List[Tuple[int, Document]]) -> List[Tuple[int, Document]]:
        if not recency_enabled:
            return items  # preserve original order
        def key(t: Tuple[int, Document]):
            idx, doc = t
            y = _extract_year(doc, recency_field)
            missing = 1 if (y is None and recency_missing_last) else 0
            # Newer first: use -y; if missing and we keep missing last, set missing=1
            sort_y = -(y if y is not None else 0)
            return (missing, sort_y, idx)
        # Stable sort based on key
        return sorted(items, key=key)

    primary_sorted = sort_by_recency(primary_list_idx)
    non_primary_sorted = sort_by_recency(non_primary_list_idx)

    required_primary = 0
    if min_primary_share > 0 and k > 0:
        required_primary = int(math.floor(k * min_primary_share))

    selected: List[Document] = []
    # First, take required_primary primaries if available
    take_primary = min(required_primary, len(primary_sorted))
    selected.extend([d for (_, d) in primary_sorted[:take_primary]])

    # Next, fill remaining slots, preferring more primaries first
    remaining = max(k - len(selected), 0)
    if remaining > 0:
        # Append more primaries
        selected.extend([d for (_, d) in primary_sorted[take_primary : take_primary + remaining]])
        remaining = max(k - len(selected), 0)
    if remaining > 0:
        # Then append non-primary
        selected.extend([d for (_, d) in non_primary_sorted[:remaining]])

    return selected[:k]


def apply_primary_corpus_bias(documents: List[Document]) -> List[Document]:
    """
    Read configuration from environment and apply reranker when enabled.

    Env vars:
      - PRIMARY_AUTHOR_BIAS_ENABLED: 'true' (default) to enable
      - PRIMARY_AUTHOR_PREFIX: 'papers/' (default)
      - PRIMARY_AUTHOR_BOOST: '2.0' (default)
      - PRIMARY_AUTHOR_MIN_SHARE: '0.8' (default)
      - PRIMARY_AUTHOR_FINAL_K: '5' (default)
    """
    enabled = (os.getenv("PRIMARY_AUTHOR_BIAS_ENABLED", "true").strip().lower() in ("1", "true", "yes"))
    if not enabled:
        return documents

    prefix = os.getenv("PRIMARY_AUTHOR_PREFIX", "papers/")
    try:
        boost = float(os.getenv("PRIMARY_AUTHOR_BOOST", "2.0"))
    except Exception:
        boost = 2.0
    try:
        min_share = float(os.getenv("PRIMARY_AUTHOR_MIN_SHARE", "0.8"))
    except Exception:
        min_share = 0.8
    try:
        final_k = int(os.getenv("PRIMARY_AUTHOR_FINAL_K", "5"))
    except Exception:
        final_k = 5

    return rerank_with_primary_weight(
        documents=documents,
        primary_prefix=prefix,
        primary_boost=boost,
        min_primary_share=min_share,
        final_k=final_k,
    )


