"""
Content quality filtering for retrieved documents.
Filters out low-quality chunks without re-embedding.
"""

import re
from typing import List
from langchain.schema import Document


def is_quality_content(text: str, min_tokens: int = 50) -> bool:
    """
    Check if content meets quality thresholds.
    
    Args:
        text: Document content to check
        min_tokens: Minimum token count (rough estimate)
    
    Returns:
        True if content passes quality checks
    """
    if not text or not text.strip():
        return False
    
    # Rough token estimate (4 chars per token)
    estimated_tokens = len(text) // 4
    if estimated_tokens < min_tokens:
        return False
    
    # Filter out reference-only content
    reference_patterns = [
        r'^References?\s*$',
        # More specific citation pattern - requires author name + year at end of line/chunk
        r'^\d+\.\s*[A-Z][a-zA-Z]+.*\b(19|20)\d{2}\s*$',  # Citation: "1. Smith et al. 2023"
        r'^\[[0-9,\s-]+\]\s*$',  # Reference numbers only (end of line)
        r'^Figure\s+\d+[\.:]\s*',  # Figure captions with colon/period
        r'^Table\s+\d+[\.:]\s*',   # Table captions with colon/period  
        r'^This figure\s+',   # Figure descriptions
        r'arXiv:\d{4}\.\d{4}',  # arXiv references (anywhere in text)
        # Additional patterns for common junk
        r'^\s*\d+\s*$',  # Just a number alone (with optional whitespace)
        r'^Page\s+\d+',  # Page numbers
        r'^–\s*\d+\s*–',  # Page ranges like "– 25 –"
    ]
    
    for pattern in reference_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE | re.MULTILINE):
            return False
    
    # Filter out content that's mostly numbers/symbols
    alpha_chars = len(re.findall(r'[a-zA-Z]', text))
    if alpha_chars < len(text) * 0.3:  # Less than 30% alphabetic
        return False
    
    # Filter out very repetitive content (headers/footers)
    lines = text.split('\n')
    unique_lines = set(line.strip() for line in lines if line.strip())
    if len(lines) > 5 and len(unique_lines) < len(lines) * 0.5:
        return False
    
    return True


def filter_quality_documents(documents: List[Document], min_tokens: int = 50) -> List[Document]:
    """
    Filter documents to remove low-quality chunks.
    
    Args:
        documents: List of retrieved documents
        min_tokens: Minimum token threshold
    
    Returns:
        Filtered list of quality documents
    """
    quality_docs = []
    
    for doc in documents:
        if is_quality_content(doc.page_content, min_tokens):
            quality_docs.append(doc)
    
    return quality_docs


def enhance_document_content(documents: List[Document]) -> List[Document]:
    """
    Enhance document content by cleaning and enriching metadata.
    
    Args:
        documents: List of documents to enhance
    
    Returns:
        Enhanced documents with cleaned content
    """
    enhanced = []
    
    for doc in documents:
        # Clean up content
        content = doc.page_content
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Add context from metadata if available
        source = doc.metadata.get("source", "")
        if source and not content.startswith("Entity:"):
            # For PDF sources, add source context
            if source.endswith('.pdf'):
                paper_name = source.split('/')[-1].replace('_', ' ').replace('.pdf', '')
                content = f"From paper: {paper_name}\n\n{content}"
        
        # Create enhanced document
        enhanced_doc = Document(
            page_content=content,
            metadata=doc.metadata
        )
        enhanced.append(enhanced_doc)
    
    return enhanced
