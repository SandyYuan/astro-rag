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
from typing import List, Tuple

from langchain.schema import Document


def _is_primary_source(doc: Document, prefix: str) -> bool:
    source = ""
    try:
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            source = str(doc.metadata.get("source", "") or "")
    except Exception:
        source = ""
    return bool(prefix) and source.startswith(prefix)


def rerank_with_primary_weight(
    documents: List[Document],
    primary_prefix: str = "papers/",
    primary_boost: float = 2.0,
    min_primary_share: float = 0.8,
    final_k: int | None = 5,
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

    primary_list: List[Document] = [d for d in documents if _is_primary_source(d, primary_prefix)]
    non_primary_list: List[Document] = [d for d in documents if not _is_primary_source(d, primary_prefix)]

    required_primary = 0
    if min_primary_share > 0 and k > 0:
        required_primary = int(math.floor(k * min_primary_share))

    selected: List[Document] = []
    # First, take required_primary primaries if available
    take_primary = min(required_primary, len(primary_list))
    selected.extend(primary_list[:take_primary])

    # Next, fill remaining slots, preferring more primaries first
    remaining = max(k - len(selected), 0)
    if remaining > 0:
        # Append more primaries
        selected.extend(primary_list[take_primary : take_primary + remaining])
        remaining = max(k - len(selected), 0)
    if remaining > 0:
        # Then append non-primary
        selected.extend(non_primary_list[:remaining])

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


