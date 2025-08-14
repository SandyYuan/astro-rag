"""
Fusion utilities for combining results from multiple retrievers.

This module implements:
- Reciprocal Rank Fusion (RRF) for combining ranked lists
- Score normalization methods
- Token budget enforcement with diversity-aware selection
- Deduplication by source
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
from langchain.schema import Document


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, Union[int, float]]]], 
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: List of ranked results, each as [(Document, rank/score), ...]
        k: RRF parameter (higher k = less aggressive fusion)
    
    Returns:
        List of (Document, fused_score) tuples, sorted by fused score descending
    """
    if not ranked_lists:
        return []
    
    # Collect all documents and their RRF scores
    doc_scores: Dict[str, Tuple[Document, float]] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            source = doc.metadata.get("source", "")
            if not source:
                # Use content hash as fallback identifier
                source = str(hash(doc.page_content))
            
            # RRF score: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            
            if source in doc_scores:
                # Sum RRF scores for documents appearing in multiple lists
                existing_doc, existing_score = doc_scores[source]
                doc_scores[source] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[source] = (doc, rrf_score)
    
    # Sort by fused score descending
    fused_results = list(doc_scores.values())
    fused_results.sort(key=lambda x: x[1], reverse=True)
    
    return fused_results


def normalize_scores(
    scored_docs: List[Tuple[Document, Optional[Union[int, float]]]], 
    method: str = "minmax"
) -> List[Tuple[Document, float]]:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scored_docs: List of (Document, score) tuples
        method: Normalization method ("minmax", "rank", "zscore")
    
    Returns:
        List of (Document, normalized_score) tuples
    """
    if not scored_docs:
        return []
    
    # Extract scores, handling None values
    scores = []
    for doc, score in scored_docs:
        if score is None:
            scores.append(0.0)
        else:
            scores.append(float(score))
    
    if method == "minmax":
        if len(set(scores)) == 1:  # All scores are the same
            normalized_scores = [1.0] * len(scores)
        else:
            min_score, max_score = min(scores), max(scores)
            normalized_scores = [
                (score - min_score) / (max_score - min_score) 
                for score in scores
            ]
    
    elif method == "rank":
        # Convert to rank-based scores (higher rank = higher score)
        # When all scores are the same (e.g., all None/0.0), preserve original order
        if len(set(scores)) == 1:
            # All scores are the same, assign descending ranks based on original order
            normalized_scores = [
                1.0 - (i / max(1, len(scores) - 1)) for i in range(len(scores))
            ]
        else:
            indexed_scores = [(score, i) for i, score in enumerate(scores)]
            indexed_scores.sort(reverse=True)  # Sort by score descending
            
            normalized_scores = [0.0] * len(scores)
            for rank, (_, original_index) in enumerate(indexed_scores):
                # Normalize rank to [0, 1], with rank 0 getting score 1.0
                normalized_scores[original_index] = 1.0 - (rank / max(1, len(scores) - 1))
    
    elif method == "zscore":
        if len(scores) <= 1:
            normalized_scores = [1.0] * len(scores)
        else:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_score = math.sqrt(variance) if variance > 0 else 1.0
            
            # Z-score normalization, then sigmoid to [0, 1]
            z_scores = [(s - mean_score) / std_score for s in scores]
            normalized_scores = [1.0 / (1.0 + math.exp(-z)) for z in z_scores]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return [(doc, norm_score) for (doc, _), norm_score in zip(scored_docs, normalized_scores)]


def deduplicate_by_source(documents: List[Document]) -> List[Document]:
    """
    Remove documents with duplicate sources, keeping the first occurrence.
    
    Args:
        documents: List of documents to deduplicate
    
    Returns:
        Deduplicated list of documents
    """
    seen_sources = set()
    deduplicated = []
    
    for doc in documents:
        source = doc.metadata.get("source")
        if source and source in seen_sources:
            continue
        
        if source:
            seen_sources.add(source)
        deduplicated.append(doc)
    
    return deduplicated


def count_tokens(text: str) -> int:
    """
    Count tokens in text using a simple approximation.
    
    For production use, this should use the actual tokenizer of the target LLM.
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Approximate token count
    """
    # Simple approximation: ~1.3 tokens per word for English text
    words = len(text.split())
    return int(words * 1.3)


def estimate_tokens(text: str) -> int:
    """
    Fast token estimation based on character count.
    
    Args:
        text: Text to estimate tokens for
    
    Returns:
        Estimated token count
    """
    # Rough approximation: ~4 characters per token for English
    return len(text) // 4


def enforce_token_budget(
    documents: List[Document], 
    budget: int,
    min_docs: int = 1,
    diversity_factor: float = 0.5
) -> List[Document]:
    """
    Select documents within token budget, prioritizing diversity.
    
    Args:
        documents: List of documents to select from
        budget: Maximum token budget
        min_docs: Minimum number of documents to return (may exceed budget)
        diversity_factor: Weight for diversity vs. relevance (0.0 = pure relevance, 1.0 = pure diversity)
    
    Returns:
        Selected documents within budget
    """
    if budget <= 0:
        return []
    
    if not documents:
        return []
    
    # Calculate token counts for all documents
    doc_tokens = [(doc, count_tokens(doc.page_content)) for doc in documents]
    
    # If we can fit all documents, return all
    total_tokens = sum(tokens for _, tokens in doc_tokens)
    if total_tokens <= budget:
        return documents
    
    # Greedy selection with diversity consideration
    selected = []
    remaining_budget = budget
    used_sources = set()
    
    # Iterative selection with dynamic scoring
    remaining_docs = doc_tokens.copy()
    
    while remaining_docs and (remaining_budget > 0 or len(selected) < min_docs):
        # Score all remaining documents based on current state
        def selection_score(doc_token_pair, position):
            doc, tokens = doc_token_pair
            source = doc.metadata.get("source", "")
            
            # Base relevance score (higher for earlier positions)
            relevance_score = 1.0 / (position + 1)
            
            # Diversity bonus for new sources (much higher weight)
            diversity_bonus = 0.0
            if source and source not in used_sources:
                diversity_bonus = 2.0  # Strong preference for new sources
            
            # Efficiency score (more content per token is better)
            content_length = len(doc.page_content)
            efficiency_score = content_length / max(tokens, 1)
            
            # Combined score with stronger diversity weighting
            combined_score = (
                (1 - diversity_factor) * relevance_score + 
                diversity_factor * diversity_bonus +
                0.05 * efficiency_score  # Smaller efficiency bonus
            )
            
            return combined_score
        
        # Score remaining documents
        scored_remaining = [
            ((doc, tokens), selection_score((doc, tokens), i))
            for i, (doc, tokens) in enumerate(remaining_docs)
        ]
        
        # Sort by selection score descending
        scored_remaining.sort(key=lambda x: x[1], reverse=True)
        
        # Select the best scoring document
        if not scored_remaining:
            break
            
        (best_doc, best_tokens), best_score = scored_remaining[0]
        
        # Check if we can afford it or need to meet minimum
        if best_tokens <= remaining_budget or len(selected) < min_docs:
            selected.append(best_doc)
            remaining_budget -= best_tokens
            
            source = best_doc.metadata.get("source")
            if source:
                used_sources.add(source)
            
            # Remove selected document from remaining
            remaining_docs = [(doc, tokens) for doc, tokens in remaining_docs 
                            if doc != best_doc]
            
            # Stop if we've used up the budget and met minimum requirements
            if remaining_budget <= 0 and len(selected) >= min_docs:
                break
        else:
            # Can't afford any more documents
            break
    
    return selected


def calculate_diversity_score(documents: List[Document]) -> float:
    """
    Calculate diversity score based on unique sources.
    
    Args:
        documents: List of documents to score
    
    Returns:
        Diversity score between 0 and 1 (1 = all unique sources)
    """
    if not documents:
        return 0.0
    
    sources = [doc.metadata.get("source", "") for doc in documents]
    unique_sources = set(source for source in sources if source)
    
    return len(unique_sources) / len(documents)
