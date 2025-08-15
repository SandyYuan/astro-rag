"""
LLM-based KG filtering component for query enrichment.

This module filters Knowledge Graph results using Gemini Flash LLM to:
1. Remove irrelevant content from KG results
2. Format relevant content for optimal vector search
3. Provide fallback when LLM fails
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document

from llm_provider import LLMProvider


logger = logging.getLogger(__name__)


class KGQueryFilter:
    """Filters and formats KG results using LLM for vector search optimization."""
    
    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider for Gemini Flash calls.
        
        Args:
            llm_provider: LLM provider configured for Gemini 2.5 Flash
        """
        self.llm_provider = llm_provider
        self.llm = llm_provider.get_llm(temperature=0.0, model_name="gemini-2.5-flash")
        self.max_kg_results = 15  # Limit to prevent token overflow
        
    def filter_and_format_kg_results(self, kg_results: List[Dict], user_query: str) -> str:
        """Main method: filter KG results and format for vector search.
        
        Args:
            kg_results: List of KG result dictionaries with page_content and metadata
            user_query: Original user query for relevance filtering
            
        Returns:
            Filtered and formatted string optimized for vector search
        """
        if not kg_results:
            logger.info("No KG results to filter")
            return ""
            
        # Limit results to prevent token overflow
        limited_results = kg_results[:self.max_kg_results]
        
        # Format KG results for LLM processing
        kg_content = self._format_kg_results_for_llm(limited_results)
        
        # Create filtering prompt
        prompt = self._create_filtering_prompt(user_query, kg_content)
        
        # Call LLM for deterministic results (temperature already set to 0.0 in __init__)
        filtered_content = self.llm(prompt)
        
        logger.info(f"LLM filtered {len(kg_results)} KG results for query: {user_query[:50]}...")
        return filtered_content.strip()
    
    def _format_kg_results_for_llm(self, kg_results: List[Dict]) -> str:
        """Format KG results into structured LLM input.
        
        Args:
            kg_results: List of KG result dictionaries
            
        Returns:
            Formatted string for LLM processing
        """
        formatted_parts = []
        
        for i, result in enumerate(kg_results, 1):
            content = result.get("page_content", "")
            entity = result.get("metadata", {}).get("entity", "Unknown")
            source = result.get("metadata", {}).get("source", "Unknown")
            
            formatted_parts.append(f"{i}. Entity: {entity}\n   Content: {content}\n   Source: {source}")
        
        return "\n\n".join(formatted_parts)
    
    def _create_filtering_prompt(self, user_query: str, kg_content: str) -> str:
        """Create optimized prompt for Gemini Flash filtering.
        
        Args:
            user_query: Original user query
            kg_content: Formatted KG content
            
        Returns:
            Prompt string for LLM filtering
        """
        prompt = f"""You are a scientific content filter. Your task is to identify and extract only the content that is directly relevant to answering the user's query.

User Query: "{user_query}"

Knowledge Graph Content:
{kg_content}

Instructions:
1. Identify which entities and content are directly relevant to the user's query
2. Remove any content that is tangentially related or off-topic
3. Format the relevant content as a concise, keyword-rich summary optimized for vector search
4. Focus on scientific concepts, methods, measurements, and findings
5. Keep the output under 100 words

Output only the filtered, relevant content suitable for vector search:"""

        return prompt
    

