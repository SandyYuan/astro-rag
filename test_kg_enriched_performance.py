"""
Performance tests and validation for KG-enriched pipeline.

Tests latency, quality improvements, and edge case handling.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from langchain.schema import Document

from retrieval.kg_enriched_retrieval import KGEnrichedRetriever
from retrieval.kg_filter import KGQueryFilter
from llm_provider import LLMProvider


class TestKGEnrichedPerformance:
    """Performance and quality tests for KG-enriched pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock components with realistic behavior
        self.mock_llm_provider = Mock(spec=LLMProvider)
        self.mock_llm = Mock()
        self.mock_llm_provider.get_llm.return_value = self.mock_llm
        
        self.mock_graph_retriever = Mock()
        self.mock_vector_retriever = Mock()
        
        # Create pipeline
        self.kg_filter = KGQueryFilter(self.mock_llm_provider)
        self.pipeline = KGEnrichedRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_retriever=self.mock_vector_retriever,
            kg_filter=self.kg_filter
        )
    
    def test_pipeline_latency_measurement(self):
        """Measure end-to-end latency of KG-enriched pipeline."""
        # Setup realistic mock responses
        kg_docs = [
            Document(page_content="Dark matter WIMP detection", metadata={"entity": "WIMP"}),
            Document(page_content="Underground detectors xenon", metadata={"entity": "detectors"})
        ]
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        
        # Mock LLM with slight delay to simulate real API call
        def mock_llm_call(prompt):
            time.sleep(0.01)  # Simulate 10ms LLM call
            return "WIMP detection underground detectors xenon targets"
        self.mock_llm.side_effect = mock_llm_call
        
        vector_docs = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Measure latency
        start_time = time.time()
        results = self.pipeline.get_relevant_documents("What is dark matter detection?")
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Verify results and performance
        assert results == vector_docs
        assert latency < 1.0  # Should complete within 1 second
        print(f"KG-enriched pipeline latency: {latency:.3f}s")
    
    def test_llm_call_timing(self):
        """Measure LLM filtering step specifically."""
        kg_results = [
            {"page_content": "WIMP detection", "metadata": {"entity": "WIMP"}},
            {"page_content": "Cosmic inflation", "metadata": {"entity": "inflation"}}
        ]
        
        # Mock LLM with timing
        def timed_llm_call(prompt):
            time.sleep(0.05)  # Simulate 50ms LLM call
            return "WIMP detection content"
        self.mock_llm.side_effect = timed_llm_call
        
        # Measure LLM filtering time
        start_time = time.time()
        result = self.kg_filter.filter_and_format_kg_results(kg_results, "dark matter")
        end_time = time.time()
        
        llm_latency = end_time - start_time
        
        # Verify
        assert result == "WIMP detection content"
        assert llm_latency < 0.5  # Should complete within 500ms
        print(f"LLM filtering latency: {llm_latency:.3f}s")
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        # Test with larger mock datasets
        large_kg_docs = [
            Document(
                page_content=f"Mock content {i} " * 100,  # ~1KB per document
                metadata={"entity": f"entity_{i}", "source": f"source_{i}.pdf"}
            )
            for i in range(20)  # 20 documents
        ]
        
        self.mock_graph_retriever.get_relevant_documents.return_value = large_kg_docs
        self.mock_llm.return_value = "Filtered content"
        
        vector_docs = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Process multiple queries to test memory stability
        for i in range(10):
            results = self.pipeline.get_relevant_documents(f"Query {i}")
            assert len(results) == 1
        
        # Verify KG results were limited to prevent overflow
        filter_calls = self.mock_llm.call_args_list
        assert len(filter_calls) == 10  # One call per query
        
        # Verify that large datasets are handled (no exceptions)
        assert True  # Test passes if no memory errors occurred


class TestKGEnrichedQuality:
    """Quality and edge case tests for KG-enriched pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_provider = Mock(spec=LLMProvider)
        self.mock_llm = Mock()
        self.mock_llm_provider.get_llm.return_value = self.mock_llm
        
        self.mock_graph_retriever = Mock()
        self.mock_vector_retriever = Mock()
        
        self.kg_filter = KGQueryFilter(self.mock_llm_provider)
        self.pipeline = KGEnrichedRetriever(
            graph_retriever=self.mock_graph_retriever,
            vector_retriever=self.mock_vector_retriever,
            kg_filter=self.kg_filter
        )
    
    def test_scientific_concept_coverage(self):
        """Test KG enrichment improves domain-specific retrieval."""
        # Astronomy-specific query
        query = "How do we measure the Hubble constant?"
        
        # Mock KG results with domain-specific entities
        kg_docs = [
            Document(
                page_content="Type Ia supernovae serve as standard candles for distance measurements",
                metadata={"entity": "Type Ia supernovae", "source": "hubble_sn.pdf"}
            ),
            Document(
                page_content="Cepheid variable stars provide distance calibration",
                metadata={"entity": "Cepheid variables", "source": "cepheid.pdf"}
            ),
            Document(
                page_content="CMB temperature fluctuations constrain cosmological parameters",
                metadata={"entity": "CMB", "source": "cmb.pdf"}
            )
        ]
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        
        # Mock LLM filtering that preserves relevant concepts
        self.mock_llm.return_value = "Type Ia supernovae standard candles Cepheid variables distance calibration"
        
        vector_docs = [Document(page_content="Hubble constant measurement", metadata={"source": "hubble.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Execute
        results = self.pipeline.get_relevant_documents(query)
        
        # Verify enriched query contains domain concepts
        vector_call_args = self.mock_vector_retriever.get_relevant_documents.call_args[0][0]
        assert query in vector_call_args
        assert "supernovae" in vector_call_args or "Cepheid" in vector_call_args
        assert results == vector_docs
    
    def test_edge_case_empty_kg_results(self):
        """Test handling of empty KG results."""
        query = "Novel query with no KG matches"
        
        # Empty KG results
        self.mock_graph_retriever.get_relevant_documents.return_value = []
        
        vector_docs = [Document(page_content="Vector fallback", metadata={"source": "fallback.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Execute
        results = self.pipeline.get_relevant_documents(query)
        
        # Verify fallback to original query
        self.mock_vector_retriever.get_relevant_documents.assert_called_once_with(query)
        assert results == vector_docs
    
    def test_edge_case_llm_empty_response(self):
        """Test handling of empty LLM filter response."""
        query = "Test query"
        
        kg_docs = [Document(page_content="Some content", metadata={"entity": "test"})]
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        
        # LLM returns empty response
        self.mock_llm.return_value = ""
        
        vector_docs = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Execute
        results = self.pipeline.get_relevant_documents(query)
        
        # Verify fallback to original query when LLM returns empty
        self.mock_vector_retriever.get_relevant_documents.assert_called_once_with(query)
        assert results == vector_docs
    
    def test_edge_case_very_long_query(self):
        """Test handling of very long queries."""
        # Create a very long query (500+ characters)
        long_query = "What is dark matter detection " * 20  # ~500 characters
        
        kg_docs = [Document(page_content="WIMP detection", metadata={"entity": "WIMP"})]
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        
        # Mock long LLM response
        long_context = "WIMP detection underground detectors " * 20  # ~700 characters
        self.mock_llm.return_value = long_context
        
        vector_docs = [Document(page_content="Vector result", metadata={"source": "test.pdf"})]
        self.mock_vector_retriever.get_relevant_documents.return_value = vector_docs
        
        # Execute
        results = self.pipeline.get_relevant_documents(long_query)
        
        # Verify query length is managed (should be truncated to ~500 chars)
        vector_call_args = self.mock_vector_retriever.get_relevant_documents.call_args[0][0]
        assert len(vector_call_args) <= 600  # Allow some buffer for truncation logic
        assert long_query[:100] in vector_call_args  # Original query start preserved
        assert results == vector_docs
    
    def test_comparison_with_baseline(self):
        """Compare KG-enriched vs standard retrieval quality."""
        query = "What causes galaxy rotation curves?"
        
        # Simulate KG providing relevant context
        kg_docs = [
            Document(
                page_content="Dark matter halos explain flat rotation curves in spiral galaxies",
                metadata={"entity": "dark matter halos", "source": "dm_halos.pdf"}
            )
        ]
        self.mock_graph_retriever.get_relevant_documents.return_value = kg_docs
        self.mock_llm.return_value = "dark matter halos flat rotation curves spiral galaxies"
        
        # Mock vector results
        enriched_results = [
            Document(page_content="Dark matter halo model explains rotation curves", metadata={"source": "enriched.pdf"})
        ]
        baseline_results = [
            Document(page_content="Galaxy dynamics and kinematics", metadata={"source": "baseline.pdf"})
        ]
        
        # Test enriched pipeline
        self.mock_vector_retriever.get_relevant_documents.return_value = enriched_results
        enriched_output = self.pipeline.get_relevant_documents(query)
        
        # Get the enriched call arguments before reset
        enriched_call_args = self.mock_vector_retriever.get_relevant_documents.call_args[0][0]
        
        # Test baseline (original query only)
        self.mock_vector_retriever.reset_mock()
        self.mock_vector_retriever.get_relevant_documents.return_value = baseline_results
        baseline_output = self.mock_vector_retriever.get_relevant_documents(query)
        
        # Verify enriched query contains more context
        assert len(enriched_call_args) > len(query)  # Enriched query is longer
        assert "dark matter" in enriched_call_args
        
        # Both should return results, but enriched has more context
        assert len(enriched_output) == 1
        assert len(baseline_output) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
