"""
KG-enriched retrieval pipeline implementation.

This module implements the sequential pipeline:
User Query → KG → LLM Filter → Query Enrichment → Vector Search → Results
"""

import logging
from typing import List, Dict
from langchain.schema import Document

from retrieval.kg_filter import KGQueryFilter
from retrieval.content_filter import filter_documents


logger = logging.getLogger(__name__)


class KGEnrichedRetriever:
    """Sequential KG-enriched retrieval pipeline."""
    
    def __init__(self, graph_retriever, vector_retriever, kg_filter: KGQueryFilter):
        """Initialize with existing retrievers and new KG filter.
        
        Args:
            graph_retriever: Neo4j GraphRetriever instance
            vector_retriever: FAISS vector retriever instance  
            kg_filter: KGQueryFilter for LLM-based filtering
        """
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        self.kg_filter = kg_filter
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Main pipeline method.
        
        Args:
            query: User query string
            
        Returns:
            List of relevant documents from vector search
        """
        logger.info(f"Starting KG-enriched retrieval for query: {query[:50]}...")
        
        # Step 1: Retrieve from Knowledge Graph
        logger.debug("Step 1: Retrieving from Knowledge Graph")
        kg_documents = self.graph_retriever.get_relevant_documents(query)
        
        if not kg_documents:
            logger.warning("No KG results found - this may indicate a problem with the knowledge graph")
            # Still proceed with original query to vector search rather than failing completely
            return self.vector_retriever.get_relevant_documents(query)
        
        logger.info(f"Retrieved {len(kg_documents)} documents from KG")
        for i, d in enumerate(kg_documents[:5], 1):
            src = d.metadata.get("source", "Unknown") if hasattr(d, "metadata") else "Unknown"
            preview = (d.page_content[:200] + "...") if hasattr(d, "page_content") and d.page_content else ""
            logger.info(f"KG[{i}] source={src} | {preview}")
        
        # Step 2: Convert to dict format and filter with LLM
        logger.debug("Step 2: Converting KG documents and filtering with LLM")
        kg_dicts = self._convert_kg_documents_to_dict(kg_documents)
        kg_context = self.kg_filter.filter_and_format_kg_results(kg_dicts, query)
        
        if not kg_context or kg_context.strip() == "":
            logger.warning("LLM filtering returned empty context - this may indicate a filtering issue")
            # Use original query rather than failing completely
            return self.vector_retriever.get_relevant_documents(query)
        
        # Step 3: Create enriched query
        logger.debug("Step 3: Creating enriched query")
        enriched_query = self._create_enriched_query(query, kg_context)
        logger.info("Enriched query (first 200 chars): %s", enriched_query[:200])
        
        # Step 4: Vector search with enriched query
        logger.debug("Step 4: Performing vector search with enriched query")
        vector_results = self.vector_retriever.get_relevant_documents(enriched_query)
        for i, d in enumerate(vector_results[:5], 1):
            src = d.metadata.get("source", "Unknown") if hasattr(d, "metadata") else "Unknown"
            preview = (d.page_content[:200] + "...") if hasattr(d, "page_content") and d.page_content else ""
            logger.info(f"VEC[{i}] source={src} | {preview}")

        # Post-filter vector results to drop citations/affiliations/boilerplate
        filtered_results = filter_documents(vector_results)
        if len(filtered_results) != len(vector_results):
            logger.info("Vector post-filter: kept %d / %d", len(filtered_results), len(vector_results))
        else:
            logger.info("Vector post-filter: no removals")
        
        logger.info(f"KG-enriched retrieval completed: {len(filtered_results)} results")
        return filtered_results
    
    def _convert_kg_documents_to_dict(self, kg_docs: List[Document]) -> List[Dict]:
        """Convert LangChain Documents to dict format for LLM processing.
        
        Args:
            kg_docs: List of LangChain Document objects
            
        Returns:
            List of dictionaries with page_content and metadata
        """
        dict_results = []
        for doc in kg_docs:
            dict_results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        return dict_results
    
    def _create_enriched_query(self, original_query: str, kg_context: str) -> str:
        """Combine original query with KG context.
        
        Args:
            original_query: User's original query
            kg_context: Filtered KG context from LLM
            
        Returns:
            Enriched query combining both sources
        """
        # Preserve original query intent while adding KG context
        if len(kg_context.strip()) == 0:
            return original_query
            
        # Create enriched query that maintains original intent
        enriched = f"{original_query} {kg_context}"
        
        # Limit total length to prevent overwhelming vector search
        max_length = 1000
        if len(enriched) > max_length:
            # If original query itself is too long, truncate it first
            if len(original_query) > max_length * 0.7:  # Reserve 30% for context
                max_query_length = int(max_length * 0.7)
                truncated_query = original_query[:max_query_length].strip()
                max_context_length = max_length - len(truncated_query) - 1
                truncated_context = kg_context[:max_context_length].strip()
                enriched = f"{truncated_query} {truncated_context}"
            else:
                # Truncate KG context while preserving original query
                max_context_length = max_length - len(original_query) - 1
                if max_context_length > 0:
                    truncated_context = kg_context[:max_context_length].strip()
                    enriched = f"{original_query} {truncated_context}"
                else:
                    # If original query is too long, return it as-is
                    enriched = original_query
        
        return enriched
