"""
Simple integration tests for KG-enriched pipeline.

Tests core integration without complex LangChain mocking.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from retrieval.kg_enriched_retrieval import KGEnrichedRetriever
from retrieval.kg_filter import KGQueryFilter
from llm_provider import LLMProvider


class TestKGEnrichedSimpleIntegration:
    """Simple integration tests for KG-enriched functionality."""
    
    def test_environment_variable_toggle(self):
        """Test that USE_KG_ENRICHED environment variable works."""
        # Test enabled
        with patch.dict(os.environ, {'USE_KG_ENRICHED': 'true'}):
            use_kg_enriched = os.environ.get("USE_KG_ENRICHED", "false").lower() == "true"
            assert use_kg_enriched is True
        
        # Test disabled
        with patch.dict(os.environ, {'USE_KG_ENRICHED': 'false'}):
            use_kg_enriched = os.environ.get("USE_KG_ENRICHED", "false").lower() == "true"
            assert use_kg_enriched is False
        
        # Test default (not set)
        with patch.dict(os.environ, {}, clear=True):
            use_kg_enriched = os.environ.get("USE_KG_ENRICHED", "false").lower() == "true"
            assert use_kg_enriched is False
    
    def test_kg_enriched_retriever_integration(self):
        """Test KG-enriched retriever with real components (mocked externals)."""
        # Mock external dependencies
        mock_llm_provider = Mock(spec=LLMProvider)
        mock_llm = Mock()
        mock_llm_provider.get_llm.return_value = mock_llm
        
        mock_graph_retriever = Mock()
        mock_vector_retriever = Mock()
        
        # Create KG filter and retriever
        kg_filter = KGQueryFilter(mock_llm_provider)
        kg_enriched_retriever = KGEnrichedRetriever(
            graph_retriever=mock_graph_retriever,
            vector_retriever=mock_vector_retriever,
            kg_filter=kg_filter
        )
        
        # Test query
        query = "What is dark matter?"
        
        # Mock KG results
        kg_docs = [Document(page_content="WIMP detection", metadata={"entity": "WIMP"})]
        mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        
        # Mock LLM response
        mock_llm.return_value = "WIMP detection underground detectors"
        
        # Mock vector results
        vector_docs = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Execute
        results = kg_enriched_retriever.get_relevant_documents(query)
        
        # Verify
        assert results == vector_docs
        mock_graph_retriever.get_relevant_documents.assert_called_once_with(query)
        mock_llm.assert_called_once()
        mock_vector_retriever.get_relevant_documents.assert_called_once()
        
        # Verify enriched query was used
        vector_call_args = mock_vector_retriever.get_relevant_documents.call_args[0][0]
        assert query in vector_call_args
        assert "WIMP" in vector_call_args
    
    def test_kg_enriched_fallback_behavior(self):
        """Test fallback behavior when KG pipeline components fail."""
        # Mock external dependencies
        mock_llm_provider = Mock(spec=LLMProvider)
        mock_llm = Mock()
        mock_llm_provider.get_llm.return_value = mock_llm
        
        mock_graph_retriever = Mock()
        mock_vector_retriever = Mock()
        
        # Create KG filter and retriever
        kg_filter = KGQueryFilter(mock_llm_provider)
        kg_enriched_retriever = KGEnrichedRetriever(
            graph_retriever=mock_graph_retriever,
            vector_retriever=mock_vector_retriever,
            kg_filter=kg_filter
        )
        
        # Test query
        query = "What is dark matter?"
        
        # Mock KG retriever failure
        mock_graph_retriever.get_relevant_documents.side_effect = Exception("KG error")
        
        # Mock vector results (fallback)
        vector_docs = [Document(page_content="Fallback result", metadata={"source": "fallback.pdf"})]
        mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Execute - should not raise exception
        results = kg_enriched_retriever.get_relevant_documents(query)
        
        # Verify fallback to original query
        assert results == vector_docs
        mock_vector_retriever.get_relevant_documents.assert_called_once_with(query)
    
    def test_kg_filter_with_real_llm_provider_interface(self):
        """Test KG filter works with real LLMProvider interface."""
        # Create mock LLM provider that behaves like the real one
        mock_llm_provider = Mock(spec=LLMProvider)
        mock_llm = Mock()
        mock_llm.return_value = "Filtered content about WIMP detection"
        mock_llm_provider.get_llm.return_value = mock_llm
        
        # Create KG filter
        kg_filter = KGQueryFilter(mock_llm_provider)
        
        # Test data
        kg_results = [
            {
                "page_content": "WIMP detection experiments",
                "metadata": {"entity": "WIMP", "source": "test1.pdf"}
            },
            {
                "page_content": "Cosmic inflation theory",
                "metadata": {"entity": "inflation", "source": "test2.pdf"}
            }
        ]
        
        # Execute filtering
        result = kg_filter.filter_and_format_kg_results(kg_results, "dark matter detection")
        
        # Verify
        assert result == "Filtered content about WIMP detection"
        mock_llm_provider.get_llm.assert_called_once_with(temperature=0.0, model_name="gemini-2.5-flash")
        mock_llm.assert_called_once()
        
        # Verify prompt contains query and KG content
        call_args = mock_llm.call_args[0][0]
        assert "dark matter detection" in call_args
        assert "WIMP detection experiments" in call_args
        assert "Cosmic inflation theory" in call_args


if __name__ == "__main__":
    pytest.main([__file__])
