"""
Test suite for KG-enriched vector search pipeline.

This module tests the new sequential retrieval approach:
User Query → KG → LLM Filter → Query Enrichment → Vector Search → Results
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from langchain.schema import Document

from retrieval.kg_filter import KGQueryFilter
from retrieval.kg_enriched_retrieval import KGEnrichedRetriever
from llm_provider import LLMProvider


class TestKGEnrichedPipelineFlow:
    """Test complete pipeline flow and integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock components
        self.mock_graph_retriever = Mock()
        self.mock_vector_retriever = Mock()
        self.mock_llm_provider = Mock(spec=LLMProvider)
        
        # Test data
        self.test_query = "What is dark matter detection?"
        
        # Mock KG results with relevant and irrelevant content
        self.mock_kg_results = [
            Document(
                page_content="WIMP detection experiments use underground detectors to search for dark matter particles.",
                metadata={"source": "http://arxiv.org/pdf/2301.12345", "entity": "WIMP detection"}
            ),
            Document(
                page_content="Cosmic microwave background radiation provides evidence for dark matter through gravitational effects.",
                metadata={"source": "http://arxiv.org/pdf/2302.67890", "entity": "cosmic microwave background"}
            ),
            Document(
                page_content="Cosmic inflation theory explains the early universe expansion and flatness problem.",
                metadata={"source": "http://arxiv.org/pdf/2303.11111", "entity": "cosmic inflation"}
            )
        ]
        
        # Mock vector search results
        self.mock_vector_results = [
            Document(
                page_content="Direct detection experiments for dark matter particles use xenon-based detectors.",
                metadata={"source": "papers/dark_matter_detection.pdf", "page": 5}
            ),
            Document(
                page_content="Indirect detection methods look for dark matter annihilation products in cosmic rays.",
                metadata={"source": "papers/indirect_detection.pdf", "page": 12}
            )
        ]

    def test_kg_enriched_pipeline_basic_flow(self):
        """Test complete pipeline: query → KG → LLM filter → vector search."""
        # Setup mocks
        self.mock_graph_retriever.get_relevant_documents.return_value = self.mock_kg_results
        self.mock_vector_retriever.get_relevant_documents.return_value = self.mock_vector_results
        
        # Mock LLM filtering response
        filtered_context = "WIMP detection experiments and cosmic microwave background evidence for dark matter"
        mock_kg_filter = Mock()
        mock_kg_filter.filter_and_format_kg_results.return_value = filtered_context
        
        # Create pipeline
        pipeline = KGEnrichedRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_retriever=self.mock_vector_retriever,
            kg_filter=mock_kg_filter
        )
        
        # Execute pipeline
        results = pipeline.get_relevant_documents(self.test_query)
        
        # Verify flow
        self.mock_graph_retriever.get_relevant_documents.assert_called_once_with(self.test_query)
        mock_kg_filter.filter_and_format_kg_results.assert_called_once()
        
        # Verify enriched query was used for vector search
        call_args = self.mock_vector_retriever.get_relevant_documents.call_args[0][0]
        assert self.test_query in call_args
        assert "WIMP detection" in call_args or "dark matter" in call_args
        
        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(doc, Document) for doc in results)
        assert len(results) == len(self.mock_vector_results)

    def test_kg_enriched_pipeline_with_empty_kg(self):
        """Test fallback when KG returns no results."""
        # Setup mocks - empty KG results
        self.mock_graph_retriever.get_relevant_documents.return_value = []
        self.mock_vector_retriever.get_relevant_documents.return_value = self.mock_vector_results
        
        mock_kg_filter = Mock()
        mock_kg_filter.filter_and_format_kg_results.return_value = ""
        
        pipeline = KGEnrichedRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_retriever=self.mock_vector_retriever,
            kg_filter=mock_kg_filter
        )
        
        # Execute pipeline
        results = pipeline.get_relevant_documents(self.test_query)
        
        # Verify fallback to original query
        self.mock_vector_retriever.get_relevant_documents.assert_called_once_with(self.test_query)
        assert len(results) == len(self.mock_vector_results)

    def test_kg_enriched_pipeline_with_llm_failure(self):
        """Test fallback when LLM filtering fails."""
        # Setup mocks
        self.mock_graph_retriever.get_relevant_documents.return_value = self.mock_kg_results
        self.mock_vector_retriever.get_relevant_documents.return_value = self.mock_vector_results
        
        # Mock LLM failure
        mock_kg_filter = Mock()
        mock_kg_filter.filter_and_format_kg_results.side_effect = Exception("LLM API error")
        
        pipeline = KGEnrichedRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_retriever=self.mock_vector_retriever,
            kg_filter=mock_kg_filter
        )
        
        # Execute pipeline - should not raise exception
        results = pipeline.get_relevant_documents(self.test_query)
        
        # Verify fallback to original query
        self.mock_vector_retriever.get_relevant_documents.assert_called_once_with(self.test_query)
        assert len(results) == len(self.mock_vector_results)


class TestLLMFiltering:
    """Test LLM-based KG content filtering."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_provider = Mock(spec=LLMProvider)
        self.mock_llm = Mock()
        self.mock_llm_provider.get_llm.return_value = self.mock_llm
        self.kg_filter = KGQueryFilter(self.mock_llm_provider)
        
        # Test data
        self.dark_matter_query = "How do we detect dark matter particles?"
        self.mixed_kg_results = [
            {
                "page_content": "WIMP detection experiments use underground detectors with xenon targets.",
                "metadata": {"entity": "WIMP detection", "source": "http://arxiv.org/pdf/2301.12345"}
            },
            {
                "page_content": "Cosmic inflation explains the flatness and horizon problems in cosmology.",
                "metadata": {"entity": "cosmic inflation", "source": "http://arxiv.org/pdf/2302.67890"}
            },
            {
                "page_content": "Dark matter halos affect galaxy rotation curves and gravitational lensing.",
                "metadata": {"entity": "dark matter halos", "source": "http://arxiv.org/pdf/2303.11111"}
            }
        ]

    def test_llm_filter_removes_irrelevant_content(self):
        """Test that cosmic inflation gets filtered out from dark matter detection query."""
        # Mock LLM response that filters out cosmic inflation
        mock_response = """Relevant content for dark matter detection:
        
        1. WIMP detection experiments use underground detectors with xenon targets.
        2. Dark matter halos affect galaxy rotation curves and gravitational lensing.
        
        The cosmic inflation content is not directly relevant to dark matter detection methods."""
        
        self.mock_llm.return_value = mock_response
        
        result = self.kg_filter.filter_and_format_kg_results(
            self.mixed_kg_results, 
            self.dark_matter_query
        )
        
        # Verify LLM was called with proper prompt
        self.mock_llm.assert_called_once()
        call_args = self.mock_llm.call_args[0][0]
        assert self.dark_matter_query in call_args
        assert "WIMP detection" in call_args
        assert "cosmic inflation" in call_args
        
        # Verify filtering result
        assert "WIMP detection" in result
        assert "gravitational lensing" in result  # This is in the mock response
        assert "cosmic inflation content is not directly relevant" in result

    def test_llm_filter_preserves_relevant_content(self):
        """Test that WIMP detection content is preserved for dark matter query."""
        relevant_kg_results = [
            {
                "page_content": "WIMP detection experiments use underground detectors.",
                "metadata": {"entity": "WIMP detection", "source": "http://arxiv.org/pdf/2301.12345"}
            },
            {
                "page_content": "Direct detection methods look for nuclear recoils from dark matter interactions.",
                "metadata": {"entity": "direct detection", "source": "http://arxiv.org/pdf/2302.67890"}
            }
        ]
        
        mock_response = """All content is relevant for dark matter detection:
        
        1. WIMP detection experiments use underground detectors.
        2. Direct detection methods look for nuclear recoils from dark matter interactions."""
        
        self.mock_llm.return_value = mock_response
        
        result = self.kg_filter.filter_and_format_kg_results(
            relevant_kg_results,
            self.dark_matter_query
        )
        
        # Verify all relevant content is preserved
        assert "WIMP detection" in result
        assert "Direct detection methods" in result  # Exact text from mock response
        assert "underground detectors" in result

    def test_llm_filter_formats_for_vector_search(self):
        """Test output format is optimized for vector search."""
        mock_response = "WIMP detection underground detectors xenon targets nuclear recoils dark matter interactions"
        self.mock_llm.return_value = mock_response
        
        result = self.kg_filter.filter_and_format_kg_results(
            self.mixed_kg_results,
            self.dark_matter_query
        )
        
        # Verify format is suitable for vector search (concise, keyword-rich)
        assert isinstance(result, str)
        assert len(result.split()) >= 5  # Has multiple keywords
        assert result == mock_response

    def test_llm_filter_fallback_on_error(self):
        """Test fallback behavior when LLM fails."""
        # Mock LLM failure
        self.mock_llm.side_effect = Exception("API error")
        
        result = self.kg_filter.filter_and_format_kg_results(
            self.mixed_kg_results,
            self.dark_matter_query
        )
        
        # Verify fallback returns simple concatenation
        assert isinstance(result, str)
        assert "WIMP detection" in result
        assert "gravitational lensing" in result  # From the fallback concatenation
        # Should include all content as fallback


class TestQueryEnrichment:
    """Test query enrichment with KG context."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock components
        self.mock_graph_retriever = Mock()
        self.mock_vector_retriever = Mock()
        self.mock_kg_filter = Mock()
        
        self.pipeline = KGEnrichedRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_retriever=self.mock_vector_retriever,
            kg_filter=self.mock_kg_filter
        )
    
    def test_query_enrichment_with_kg_context(self):
        """Test enriched query contains both original query and KG context."""
        original_query = "What is dark matter?"
        kg_context = "WIMP detection underground detectors xenon targets"
        
        # Mock KG retrieval
        kg_docs = [Document(page_content="WIMP content", metadata={"entity": "WIMP"})]
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        
        # Mock KG filtering
        self.mock_kg_filter.filter_and_format_kg_results.return_value = kg_context
        
        # Mock vector search results
        vector_results = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_results
        
        # Execute pipeline
        results = self.pipeline.get_relevant_documents(original_query)
        
        # Verify enriched query was used
        vector_call_args = self.mock_vector_retriever.get_relevant_documents.call_args[0][0]
        assert original_query in vector_call_args
        assert "WIMP" in vector_call_args or "xenon" in vector_call_args
        assert results == vector_results
    
    def test_query_enrichment_fallback(self):
        """Test original query is used when no KG context available."""
        original_query = "What is dark matter?"
        
        # Mock empty KG results
        self.mock_graph_retriever.get_relevant_documents.return_value = []
        
        # Mock vector search results
        vector_results = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_results
        
        # Execute pipeline
        results = self.pipeline.get_relevant_documents(original_query)
        
        # Verify original query was used directly
        self.mock_vector_retriever.get_relevant_documents.assert_called_once_with(original_query)
        assert results == vector_results
        
    def test_document_format_conversion(self):
        """Test conversion from LangChain Documents to dict format."""
        # Mock KG documents
        kg_docs = [
            Document(
                page_content="WIMP detection content",
                metadata={"entity": "WIMP", "source": "http://arxiv.org/pdf/123"}
            ),
            Document(
                page_content="Dark matter halo content", 
                metadata={"entity": "dark matter halo", "source": "http://arxiv.org/pdf/456"}
            )
        ]
        
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        self.mock_kg_filter.filter_and_format_kg_results.return_value = "filtered content"
        self.mock_vector_retriever.get_relevant_documents.return_value = []
        
        # Execute pipeline
        self.pipeline.get_relevant_documents("test query")
        
        # Verify conversion to dict format
        filter_call_args = self.mock_kg_filter.filter_and_format_kg_results.call_args[0][0]
        assert isinstance(filter_call_args, list)
        assert len(filter_call_args) == 2
        assert filter_call_args[0]["page_content"] == "WIMP detection content"
        assert filter_call_args[0]["metadata"]["entity"] == "WIMP"
        
    def test_error_handling_and_fallback(self):
        """Test error handling at each pipeline step."""
        original_query = "What is dark matter?"
        
        # Mock KG retriever failure
        self.mock_graph_retriever.get_relevant_documents.side_effect = Exception("KG error")
        
        # Mock vector search results
        vector_results = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_results
        
        # Execute pipeline - should not raise exception
        results = self.pipeline.get_relevant_documents(original_query)
        
        # Verify fallback to original query
        self.mock_vector_retriever.get_relevant_documents.assert_called_once_with(original_query)
        assert results == vector_results


# Test data constants for reuse across test files
TEST_QUERIES = {
    "dark_matter": "What is dark matter detection?",
    "cosmic_inflation": "How does cosmic inflation work?",
    "gravitational_waves": "What are gravitational wave detectors?",
    "galaxy_formation": "How do galaxies form and evolve?"
}

MOCK_KG_RESULTS_RELEVANT = [
    Document(
        page_content="WIMP detection experiments use underground detectors to search for dark matter particles.",
        metadata={"source": "http://arxiv.org/pdf/2301.12345", "entity": "WIMP detection"}
    ),
    Document(
        page_content="Direct detection methods look for nuclear recoils from dark matter interactions.",
        metadata={"source": "http://arxiv.org/pdf/2302.67890", "entity": "direct detection"}
    )
]

MOCK_KG_RESULTS_MIXED = [
    Document(
        page_content="WIMP detection experiments use underground detectors with xenon targets.",
        metadata={"source": "http://arxiv.org/pdf/2301.12345", "entity": "WIMP detection"}
    ),
    Document(
        page_content="Cosmic inflation explains the flatness and horizon problems in cosmology.",
        metadata={"source": "http://arxiv.org/pdf/2302.67890", "entity": "cosmic inflation"}
    ),
    Document(
        page_content="Galaxy rotation curves provide evidence for dark matter halos.",
        metadata={"source": "http://arxiv.org/pdf/2303.11111", "entity": "dark matter halos"}
    )
]

if __name__ == "__main__":
    pytest.main([__file__])
