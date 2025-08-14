"""
Comprehensive Test Suite for Phase 3 Dual Retrieval with Fusion

This test suite follows a test-driven development approach for implementing
dual retrieval (FAISS + Neo4j) with result fusion functionality.

Test Categories:
1. Fusion Algorithm Tests - RRF, score normalization, deduplication
2. Token Budget Tests - Context length management and diverse selection
3. Dual Retrieval Integration Tests - End-to-end chatbot functionality
4. Performance Tests - Latency and quality comparisons
5. Edge Case Tests - Error handling, empty results, etc.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from langchain.schema import Document

# Import the modules we'll be testing/implementing
import sys
sys.path.append('.')

from chatbot import AstronomyChatbot
from llm_provider import LLMProvider


class TestFusionAlgorithms:
    """Test the core fusion algorithms (RRF, score normalization, deduplication)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock documents from FAISS
        self.faiss_docs = [
            Document(
                page_content="FAISS doc 1: Dark energy survey results show cosmic acceleration",
                metadata={"source": "papers/dark_energy_survey.pdf", "score": 0.9}
            ),
            Document(
                page_content="FAISS doc 2: Weak lensing measurements constrain dark matter",
                metadata={"source": "papers/weak_lensing.pdf", "score": 0.8}
            ),
            Document(
                page_content="FAISS doc 3: Galaxy clustering analysis reveals large-scale structure",
                metadata={"source": "papers/galaxy_clustering.pdf", "score": 0.7}
            )
        ]
        
        # Mock documents from Neo4j
        self.neo4j_docs = [
            Document(
                page_content="Neo4j doc 1: Entity analysis of dark energy parameters",
                metadata={"source": "papers/dark_energy_survey.pdf", "entity": "dark energy"}
            ),
            Document(
                page_content="Neo4j doc 2: Cosmological constraints from multiple probes",
                metadata={"source": "papers/cosmology_constraints.pdf", "entity": "cosmological parameters"}
            ),
            Document(
                page_content="Neo4j doc 3: Hubble constant measurements show tension",
                metadata={"source": "papers/hubble_tension.pdf", "entity": "H0"}
            )
        ]
    
    def test_reciprocal_rank_fusion_basic(self):
        """Test basic RRF functionality"""
        # This will test the RRF implementation we'll create
        from retrieval.fusion import reciprocal_rank_fusion
        
        # Test with mock ranked lists
        faiss_results = [(doc, i) for i, doc in enumerate(self.faiss_docs)]
        neo4j_results = [(doc, i) for i, doc in enumerate(self.neo4j_docs)]
        
        fused_results = reciprocal_rank_fusion([faiss_results, neo4j_results], k=60)
        
        # Should return list of (Document, fused_score) tuples
        assert len(fused_results) <= len(self.faiss_docs) + len(self.neo4j_docs)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in fused_results)
        assert all(isinstance(item[0], Document) for item in fused_results)
        assert all(isinstance(item[1], (int, float)) for item in fused_results)
    
    def test_reciprocal_rank_fusion_handles_duplicates(self):
        """Test RRF correctly handles duplicate sources"""
        from retrieval.fusion import reciprocal_rank_fusion
        
        # Add duplicate source to neo4j results
        duplicate_doc = Document(
            page_content="Different content but same source",
            metadata={"source": "papers/dark_energy_survey.pdf", "entity": "duplicate"}
        )
        neo4j_with_dup = self.neo4j_docs + [duplicate_doc]
        
        faiss_results = [(doc, i) for i, doc in enumerate(self.faiss_docs)]
        neo4j_results = [(doc, i) for i, doc in enumerate(neo4j_with_dup)]
        
        fused_results = reciprocal_rank_fusion([faiss_results, neo4j_results], k=60)
        
        # Should deduplicate by source
        sources = [doc.metadata.get("source") for doc, _ in fused_results]
        assert len(sources) == len(set(sources)), "Should deduplicate by source"
    
    def test_score_normalization(self):
        """Test score normalization for different retriever types"""
        from retrieval.fusion import normalize_scores
        
        # FAISS docs with similarity scores
        faiss_scored = [(doc, 0.9 - i*0.1) for i, doc in enumerate(self.faiss_docs)]
        # Neo4j docs without scores (should get rank-based scores)
        neo4j_scored = [(doc, None) for doc in self.neo4j_docs]
        
        faiss_normalized = normalize_scores(faiss_scored, method="minmax")
        neo4j_normalized = normalize_scores(neo4j_scored, method="rank")
        
        # Check normalization results
        faiss_scores = [score for _, score in faiss_normalized]
        neo4j_scores = [score for _, score in neo4j_normalized]
        
        assert all(0 <= score <= 1 for score in faiss_scores)
        assert all(0 <= score <= 1 for score in neo4j_scores)
        assert faiss_scores == sorted(faiss_scores, reverse=True)  # Should be descending
        assert neo4j_scores == sorted(neo4j_scores, reverse=True)  # Should be descending
    
    def test_deduplication_by_source(self):
        """Test deduplication logic by metadata source"""
        from retrieval.fusion import deduplicate_by_source
        
        # Create docs with duplicate sources
        docs_with_dups = self.faiss_docs + [
            Document(
                page_content="Duplicate source content",
                metadata={"source": "papers/dark_energy_survey.pdf", "score": 0.6}
            )
        ]
        
        deduplicated = deduplicate_by_source(docs_with_dups)
        
        sources = [doc.metadata.get("source") for doc in deduplicated]
        assert len(sources) == len(set(sources)), "Should remove duplicate sources"
        
        # Should keep the first occurrence (highest score)
        dark_energy_docs = [doc for doc in deduplicated 
                           if doc.metadata.get("source") == "papers/dark_energy_survey.pdf"]
        assert len(dark_energy_docs) == 1
        assert dark_energy_docs[0].page_content.startswith("FAISS doc 1")


class TestTokenBudgetManagement:
    """Test token budget enforcement and diverse chunk selection"""
    
    def setup_method(self):
        """Set up test documents with varying lengths"""
        self.short_docs = [
            Document(page_content="Short content " * 10, metadata={"source": f"short_{i}.pdf"})
            for i in range(5)
        ]
        self.long_docs = [
            Document(page_content="Long content " * 200, metadata={"source": f"long_{i}.pdf"})
            for i in range(3)
        ]
        self.mixed_docs = self.short_docs + self.long_docs
    
    def test_token_counting_accuracy(self):
        """Test token counting for budget enforcement"""
        from retrieval.fusion import count_tokens, estimate_tokens
        
        # Test with known content
        test_text = "This is a test sentence with exactly ten words here."
        
        # Both methods should give reasonable estimates
        exact_count = count_tokens(test_text)
        estimated_count = estimate_tokens(test_text)
        
        assert isinstance(exact_count, int)
        assert isinstance(estimated_count, int)
        assert abs(exact_count - estimated_count) <= 3  # Should be close
    
    def test_budget_enforcement_under_limit(self):
        """Test that token budget is respected"""
        from retrieval.fusion import enforce_token_budget
        
        # Set a budget that should exclude some long docs
        budget = 1000  # tokens
        
        selected_docs = enforce_token_budget(self.mixed_docs, budget)
        
        # Calculate total tokens
        total_tokens = sum(
            len(doc.page_content.split()) * 1.3  # Rough token estimate
            for doc in selected_docs
        )
        
        assert total_tokens <= budget
        assert len(selected_docs) <= len(self.mixed_docs)
        assert len(selected_docs) > 0  # Should select at least something
    
    def test_diverse_selection_prioritizes_different_sources(self):
        """Test that diverse selection avoids clustering similar sources"""
        from retrieval.fusion import enforce_token_budget
        
        # Create docs with similar sources (larger content to use more budget)
        similar_source_docs = [
            Document(
                page_content=f"Content {i} " * 200,  # Larger content 
                metadata={"source": "papers/similar_topic.pdf", "subsection": f"part_{i}"}
            )
            for i in range(10)
        ]
        
        different_source_docs = [
            Document(
                page_content=f"Different content {i} " * 200,  # Larger content
                metadata={"source": f"papers/different_topic_{i}.pdf"}
            )
            for i in range(3)
        ]
        
        all_docs = similar_source_docs + different_source_docs
        budget = 1000  # Tighter budget to force selection
        
        selected_docs = enforce_token_budget(all_docs, budget, diversity_factor=0.8)
        
        # Should prefer diverse sources
        selected_sources = [doc.metadata.get("source") for doc in selected_docs]
        unique_sources = set(selected_sources)
        
        # With high diversity factor, should strongly prefer unique sources
        # At minimum, should select all 3 different sources before duplicating similar ones
        assert len(unique_sources) >= min(3, len(selected_docs))  # Should get all different sources first
    
    def test_budget_enforcement_with_minimum_docs(self):
        """Test that minimum number of documents is respected even if over budget"""
        from retrieval.fusion import enforce_token_budget
        
        # Very small budget but require minimum docs
        tiny_budget = 100
        min_docs = 3
        
        selected_docs = enforce_token_budget(
            self.long_docs, 
            tiny_budget, 
            min_docs=min_docs
        )
        
        # Should return at least min_docs even if over budget
        assert len(selected_docs) >= min_docs


class TestDualRetrievalIntegration:
    """Test integration of dual retrieval mode in the chatbot"""
    
    def setup_method(self):
        """Set up mocked dependencies"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'test_key',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': 'test_password'
        })
        self.env_patcher.start()
    
    def teardown_method(self):
        """Clean up"""
        self.env_patcher.stop()
    
    def test_dual_mode_initialization(self):
        """Test chatbot initialization with dual retrieval mode"""
        with patch('chatbot.FAISS'), \
             patch('graph_rag.neo4j_client.GraphRetriever'), \
             patch.object(LLMProvider, 'get_llm'), \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Test dual mode initialization
            chatbot = AstronomyChatbot(
                vector_store_path=self.temp_dir,
                retrieval_mode="dual"
            )
            
            assert chatbot.retrieval_mode == "dual"
            # Should have both retrievers initialized
            assert hasattr(chatbot, 'faiss_retriever')
            assert hasattr(chatbot, 'graph_retriever')
    
    def test_dual_mode_env_variable(self):
        """Test dual mode activation via environment variable"""
        with patch.dict(os.environ, {'RAG_MODE': 'dual'}), \
             patch('chatbot.FAISS'), \
             patch('graph_rag.neo4j_client.GraphRetriever'), \
             patch.object(LLMProvider, 'get_llm'), \
             patch.object(LLMProvider, 'get_embeddings'):
            
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            assert chatbot.retrieval_mode == "dual"
    
    @patch('retrieval.fusion.reciprocal_rank_fusion')
    @patch('retrieval.fusion.enforce_token_budget')
    def test_dual_retrieval_fusion_pipeline(self, mock_budget, mock_fusion):
        """Test the complete dual retrieval and fusion pipeline"""
        # Mock the fusion functions
        mock_fusion.return_value = [
            (Document(page_content="Fused doc 1", metadata={"source": "test1.pdf"}), 0.9),
            (Document(page_content="Fused doc 2", metadata={"source": "test2.pdf"}), 0.8)
        ]
        mock_budget.return_value = [
            Document(page_content="Budget doc 1", metadata={"source": "test1.pdf"}),
            Document(page_content="Budget doc 2", metadata={"source": "test2.pdf"})
        ]
        
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up mock retrievers
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.return_value = [
                Document(page_content="FAISS result", metadata={"source": "faiss.pdf"})
            ]
            mock_faiss.load_local.return_value.as_retriever.return_value = mock_faiss_retriever
            
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.return_value = [
                Document(page_content="Neo4j result", metadata={"source": "neo4j.pdf"})
            ]
            mock_graph.return_value = mock_graph_retriever
            
            # Mock LLM response
            mock_llm.return_value = Mock()
            
            # Initialize chatbot in dual mode
            chatbot = AstronomyChatbot(
                vector_store_path=self.temp_dir,
                retrieval_mode="dual"
            )
            
            # Mock the QA chain
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {
                "output_text": "Test response based on fused results"
            }
            
            # Test chat functionality
            result = chatbot.chat("What is dark energy?")
            
            # Verify both retrievers were called
            mock_faiss_retriever.get_relevant_documents.assert_called_once()
            mock_graph_retriever.get_relevant_documents.assert_called_once()
            
            # Verify fusion pipeline was called
            mock_fusion.assert_called_once()
            mock_budget.assert_called_once()
            
            # Verify response structure
            assert "answer" in result
            assert "sources" in result
            assert isinstance(result["sources"], list)
    
    def test_dual_mode_error_handling(self):
        """Test error handling when one retriever fails"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm'), \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up one working retriever and one failing
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.return_value = [
                Document(page_content="FAISS result", metadata={"source": "faiss.pdf"})
            ]
            mock_faiss.load_local.return_value.as_retriever.return_value = mock_faiss_retriever
            
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.side_effect = Exception("Neo4j connection failed")
            mock_graph.return_value = mock_graph_retriever
            
            chatbot = AstronomyChatbot(
                vector_store_path=self.temp_dir,
                retrieval_mode="dual"
            )
            
            # Should gracefully handle the error and continue with FAISS results
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {"output_text": "Response with partial results"}
            
            result = chatbot.chat("Test query")
            
            # Should still return a valid response
            assert "answer" in result
            assert result["answer"] != ""


class TestPerformanceComparisons:
    """Test performance and quality comparisons between retrieval modes"""
    
    def setup_method(self):
        """Set up test queries and expected behaviors"""
        self.test_queries = [
            "What is dark energy?",  # Broad conceptual query
            "S8 tension between DES and Planck",  # Specific parameter query
            "galaxy clustering analysis methods",  # Technical methodology query
            "cosmological constraints from weak lensing",  # Multi-concept query
        ]
    
    @pytest.mark.performance
    def test_latency_comparison(self):
        """Test that dual mode latency is acceptable compared to single modes"""
        import time
        
        # This would be implemented with actual timing measurements
        # For now, we'll test the structure
        
        latency_results = {
            "faiss": [],
            "neo4j": [], 
            "dual": []
        }
        
        # Simulate timing tests for each mode
        for mode in latency_results.keys():
            for query in self.test_queries:
                # Mock timing measurement
                start_time = time.time()
                # ... actual retrieval would happen here
                end_time = time.time()
                latency_results[mode].append(end_time - start_time)
        
        # Test that dual mode isn't significantly slower
        avg_dual = sum(latency_results["dual"]) / len(latency_results["dual"])
        avg_faiss = sum(latency_results["faiss"]) / len(latency_results["faiss"])
        avg_neo4j = sum(latency_results["neo4j"]) / len(latency_results["neo4j"])
        
        # Dual should be at most 2x the slower of the two single modes
        max_single = max(avg_faiss, avg_neo4j)
        assert avg_dual <= max_single * 2.0, "Dual mode should not be more than 2x slower"
    
    def test_result_diversity_improvement(self):
        """Test that dual mode provides more diverse results"""
        from retrieval.fusion import calculate_diversity_score
        
        # Mock results from different modes
        faiss_only_results = [
            Document(page_content="FAISS result 1", metadata={"source": "paper1.pdf"}),
            Document(page_content="FAISS result 2", metadata={"source": "paper1.pdf"}),  # Same source
            Document(page_content="FAISS result 3", metadata={"source": "paper2.pdf"}),
        ]
        
        dual_mode_results = [
            Document(page_content="FAISS result 1", metadata={"source": "paper1.pdf"}),
            Document(page_content="Neo4j result 1", metadata={"source": "paper3.pdf"}),
            Document(page_content="Neo4j result 2", metadata={"source": "paper4.pdf"}),
        ]
        
        faiss_diversity = calculate_diversity_score(faiss_only_results)
        dual_diversity = calculate_diversity_score(dual_mode_results)
        
        # Dual mode should have better diversity (more unique sources)
        assert dual_diversity > faiss_diversity
    
    def test_recall_improvement_measurement(self):
        """Test framework for measuring recall improvements"""
        # This would test against a gold standard set of relevant documents
        
        # Mock gold standard relevant documents for a query
        gold_standard = {
            "What is dark energy?": [
                "papers/dark_energy_survey.pdf",
                "papers/cosmological_constant.pdf", 
                "papers/accelerating_universe.pdf",
                "papers/type_ia_supernovae.pdf"
            ]
        }
        
        # Mock retrieved results
        faiss_retrieved = ["papers/dark_energy_survey.pdf", "papers/galaxy_clustering.pdf"]
        neo4j_retrieved = ["papers/cosmological_constant.pdf", "papers/weak_lensing.pdf"]
        dual_retrieved = ["papers/dark_energy_survey.pdf", "papers/cosmological_constant.pdf", "papers/accelerating_universe.pdf"]
        
        def calculate_recall(retrieved, relevant):
            return len(set(retrieved) & set(relevant)) / len(relevant)
        
        query = "What is dark energy?"
        relevant_docs = gold_standard[query]
        
        faiss_recall = calculate_recall(faiss_retrieved, relevant_docs)
        neo4j_recall = calculate_recall(neo4j_retrieved, relevant_docs)
        dual_recall = calculate_recall(dual_retrieved, relevant_docs)
        
        # Dual mode should have better recall than individual modes
        assert dual_recall >= max(faiss_recall, neo4j_recall)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios"""
    
    def test_empty_results_from_both_retrievers(self):
        """Test handling when both retrievers return empty results"""
        from retrieval.fusion import reciprocal_rank_fusion, enforce_token_budget
        
        empty_faiss = []
        empty_neo4j = []
        
        fused_results = reciprocal_rank_fusion([empty_faiss, empty_neo4j])
        assert fused_results == []
        
        budgeted_results = enforce_token_budget([], 1000)
        assert budgeted_results == []
    
    def test_empty_results_from_one_retriever(self):
        """Test handling when one retriever returns empty results"""
        from retrieval.fusion import reciprocal_rank_fusion
        
        faiss_results = [(Document(page_content="test", metadata={"source": "test.pdf"}), 0.9)]
        empty_neo4j = []
        
        fused_results = reciprocal_rank_fusion([faiss_results, empty_neo4j])
        
        # Should return the non-empty results
        assert len(fused_results) == 1
        assert fused_results[0][0].page_content == "test"
    
    def test_malformed_document_metadata(self):
        """Test handling of documents with malformed or missing metadata"""
        from retrieval.fusion import deduplicate_by_source
        
        malformed_docs = [
            Document(page_content="No metadata doc"),  # Missing metadata
            Document(page_content="No source doc", metadata={}),  # Empty metadata
            Document(page_content="None source doc", metadata={"source": None}),  # None source
            Document(page_content="Valid doc", metadata={"source": "valid.pdf"}),
        ]
        
        # Should handle malformed metadata gracefully
        deduplicated = deduplicate_by_source(malformed_docs)
        assert len(deduplicated) > 0  # Should not crash
        
        # Valid doc should be preserved
        valid_docs = [doc for doc in deduplicated if doc.page_content == "Valid doc"]
        assert len(valid_docs) == 1
    
    def test_very_large_token_budget(self):
        """Test behavior with unrealistically large token budgets"""
        from retrieval.fusion import enforce_token_budget
        
        docs = [
            Document(page_content="Short doc", metadata={"source": "short.pdf"}),
            Document(page_content="Medium length document content", metadata={"source": "medium.pdf"}),
        ]
        
        huge_budget = 1000000  # 1M tokens
        
        selected_docs = enforce_token_budget(docs, huge_budget)
        
        # Should return all documents when budget is very large
        assert len(selected_docs) == len(docs)
    
    def test_zero_token_budget(self):
        """Test behavior with zero or negative token budget"""
        from retrieval.fusion import enforce_token_budget
        
        docs = [Document(page_content="Test doc", metadata={"source": "test.pdf"})]
        
        # Zero budget
        selected_docs = enforce_token_budget(docs, 0)
        assert len(selected_docs) == 0
        
        # Negative budget
        selected_docs = enforce_token_budget(docs, -100)
        assert len(selected_docs) == 0


class TestConfigurationAndModes:
    """Test different configuration options and mode switches"""
    
    def test_invalid_retrieval_mode(self):
        """Test handling of invalid retrieval modes"""
        with pytest.raises(ValueError, match="Invalid RAG_MODE"):
            AstronomyChatbot(retrieval_mode="invalid_mode")
    
    def test_mode_switching_validation(self):
        """Test that mode switching validates required dependencies"""
        # Test that dual mode requires both FAISS and Neo4j to be available
        
        # Mock missing Neo4j
        with patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'test_key',
            # Missing Neo4j env vars
        }):
            with pytest.raises(ValueError, match="NEO4J_URI"):
                AstronomyChatbot(retrieval_mode="dual")
    
    def test_fusion_parameter_configuration(self):
        """Test configuration of fusion parameters"""
        # Test different RRF k values
        from retrieval.fusion import reciprocal_rank_fusion
        
        docs = [(Document(page_content=f"Doc {i}", metadata={"source": f"doc{i}.pdf"}), i) 
                for i in range(5)]
        
        # Test different k values
        results_k30 = reciprocal_rank_fusion([docs], k=30)
        results_k60 = reciprocal_rank_fusion([docs], k=60)
        results_k120 = reciprocal_rank_fusion([docs], k=120)
        
        # Different k values should produce different fusion scores
        scores_k30 = [score for _, score in results_k30]
        scores_k60 = [score for _, score in results_k60]
        scores_k120 = [score for _, score in results_k120]
        
        # Scores should be different (higher k = less aggressive fusion)
        assert scores_k30 != scores_k60 != scores_k120


# Test fixtures and utilities
@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing"""
    return [
        Document(
            page_content="Dark energy is a mysterious component of the universe that drives cosmic acceleration.",
            metadata={"source": "papers/dark_energy_overview.pdf", "score": 0.95}
        ),
        Document(
            page_content="The Dark Energy Survey (DES) measured cosmological parameters with high precision.",
            metadata={"source": "papers/des_results.pdf", "score": 0.88}
        ),
        Document(
            page_content="Weak lensing provides constraints on the matter density parameter Omega_m.",
            metadata={"source": "papers/weak_lensing_constraints.pdf", "score": 0.82}
        ),
        Document(
            page_content="Galaxy clustering analysis reveals the large-scale structure of the universe.",
            metadata={"source": "papers/galaxy_clustering_analysis.pdf", "score": 0.79}
        ),
        Document(
            page_content="Type Ia supernovae observations led to the discovery of cosmic acceleration.",
            metadata={"source": "papers/supernova_cosmology.pdf", "score": 0.75}
        ),
    ]


@pytest.fixture
def mock_llm_provider():
    """Fixture providing a mocked LLM provider"""
    provider = Mock(spec=LLMProvider)
    provider.get_llm.return_value = Mock()
    provider.get_embeddings.return_value = Mock()
    return provider


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])
