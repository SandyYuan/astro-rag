"""
Integration tests for dual retrieval mode.

These tests verify that the complete dual retrieval pipeline works end-to-end,
including both FAISS and Neo4j retrieval, fusion, and response generation.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

# Import the modules we'll be testing
import sys
sys.path.append('.')

from chatbot import AstronomyChatbot
from llm_provider import LLMProvider


class TestDualModeIntegration:
    """Integration tests for dual retrieval mode"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'test_key',
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USER': 'neo4j',
            'NEO4J_PASSWORD': 'test_password',
            'RAG_MODE': 'dual',
            'FUSION_TOKEN_BUDGET': '2000',
            'FUSION_DIVERSITY_FACTOR': '0.7'
        })
        self.env_patcher.start()
        
        # Sample documents for testing
        self.sample_faiss_docs = [
            Document(
                page_content="Dark energy survey results show accelerating universe expansion.",
                metadata={"source": "papers/dark_energy_survey.pdf", "score": 0.92}
            ),
            Document(
                page_content="Weak lensing measurements constrain matter density parameter.",
                metadata={"source": "papers/weak_lensing.pdf", "score": 0.87}
            ),
            Document(
                page_content="Galaxy clustering provides insights into large-scale structure.",
                metadata={"source": "papers/galaxy_clustering.pdf", "score": 0.81}
            )
        ]
        
        self.sample_neo4j_docs = [
            Document(
                page_content="Entity: S8\n- DES Y3 measured S8 = 0.792±0.012 from cosmic shear\n- Planck CMB predicts S8 = 0.834±0.016\n- 2.3σ tension between measurements",
                metadata={"source": "http://arxiv.org/pdf/2207.05766v4", "entity": "S8"}
            ),
            Document(
                page_content="Entity: Dark Energy\n- Constitutes ~68% of universe energy density\n- Drives accelerating expansion\n- Consistent with cosmological constant",
                metadata={"source": "papers/cosmology_overview.pdf", "entity": "Dark Energy"}
            )
        ]
    
    def teardown_method(self):
        """Clean up"""
        self.env_patcher.stop()
    
    def _create_mock_llm(self):
        """Helper to create a properly mocked LLM"""
        from langchain_core.runnables import Runnable
        return Mock(spec=Runnable)
    
    def test_dual_mode_initialization_success(self):
        """Test successful initialization of dual mode"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Mock successful FAISS loading
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = Mock()
            
            # Mock successful Neo4j connection
            mock_graph.return_value = Mock()
            
            # Mock LLM with Runnable interface
            from langchain_core.runnables import Runnable
            mock_llm_instance = Mock(spec=Runnable)
            mock_llm.return_value = mock_llm_instance
            
            # Initialize chatbot
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            
            # Verify dual mode setup
            assert chatbot.retrieval_mode == "dual"
            assert hasattr(chatbot, 'faiss_retriever')
            assert hasattr(chatbot, 'graph_retriever')
            assert chatbot.retriever is None  # No single retriever in dual mode
    
    def test_dual_retrieval_fusion_pipeline(self):
        """Test the complete dual retrieval and fusion pipeline"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up mock retrievers
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.return_value = self.sample_faiss_docs
            
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.return_value = self.sample_neo4j_docs
            
            # Mock FAISS setup
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = mock_faiss_retriever
            
            # Mock Neo4j setup
            mock_graph.return_value = mock_graph_retriever
            
            # Mock LLM with Runnable interface
            from langchain_core.runnables import Runnable
            mock_llm_instance = Mock(spec=Runnable)
            mock_llm.return_value = mock_llm_instance
            
            # Initialize chatbot
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            
            # Mock the QA chain
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {
                "output_text": "Based on the fused results, dark energy drives cosmic acceleration and shows tension in S8 measurements between DES and Planck."
            }
            
            # Test chat functionality
            result = chatbot.chat("What is the S8 tension?")
            
            # Verify both retrievers were called
            mock_faiss_retriever.get_relevant_documents.assert_called_once()
            mock_graph_retriever.get_relevant_documents.assert_called_once()
            
            # Verify QA chain was called with fused documents
            qa_call_args = chatbot.qa_chain.invoke.call_args[0][0]
            assert "input_documents" in qa_call_args
            fused_docs = qa_call_args["input_documents"]
            
            # Should have documents from both sources (after fusion and budget enforcement)
            assert len(fused_docs) > 0
            assert len(fused_docs) <= len(self.sample_faiss_docs) + len(self.sample_neo4j_docs)
            
            # Verify response structure
            assert "answer" in result
            assert "sources" in result
            assert isinstance(result["sources"], list)
            assert len(result["sources"]) > 0
    
    def test_dual_mode_error_handling_faiss_failure(self):
        """Test graceful handling when FAISS retrieval fails"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up failing FAISS retriever
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.side_effect = Exception("FAISS index corrupted")
            
            # Set up working Neo4j retriever
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.return_value = self.sample_neo4j_docs
            
            # Mock setup
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = mock_faiss_retriever
            mock_graph.return_value = mock_graph_retriever
            
            # Mock LLM
            mock_llm.return_value = self._create_mock_llm()
            
            # Initialize chatbot
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {"output_text": "Response from Neo4j only"}
            
            # Test chat functionality
            result = chatbot.chat("Test query")
            
            # Should still work with Neo4j results only
            assert "answer" in result
            assert result["answer"] != ""
            
            # Verify both retrievers were attempted
            mock_faiss_retriever.get_relevant_documents.assert_called_once()
            mock_graph_retriever.get_relevant_documents.assert_called_once()
    
    def test_dual_mode_error_handling_neo4j_failure(self):
        """Test graceful handling when Neo4j retrieval fails"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up working FAISS retriever
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.return_value = self.sample_faiss_docs
            
            # Set up failing Neo4j retriever
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.side_effect = Exception("Neo4j connection failed")
            
            # Mock setup
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = mock_faiss_retriever
            mock_graph.return_value = mock_graph_retriever
            
            # Mock LLM
            mock_llm.return_value = self._create_mock_llm()
            
            # Initialize chatbot
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {"output_text": "Response from FAISS only"}
            
            # Test chat functionality
            result = chatbot.chat("Test query")
            
            # Should still work with FAISS results only
            assert "answer" in result
            assert result["answer"] != ""
    
    def test_dual_mode_both_retrievers_fail(self):
        """Test handling when both retrievers fail"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up both retrievers to fail
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.side_effect = Exception("FAISS failed")
            
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.side_effect = Exception("Neo4j failed")
            
            # Mock setup
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = mock_faiss_retriever
            mock_graph.return_value = mock_graph_retriever
            
            # Mock LLM
            mock_llm.return_value = self._create_mock_llm()
            
            # Initialize chatbot
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {"output_text": "No context available"}
            
            # Test chat functionality
            result = chatbot.chat("Test query")
            
            # Should return error response gracefully
            assert "answer" in result
            # The chat method should handle this gracefully and return an error message
    
    def test_fusion_parameters_from_environment(self):
        """Test that fusion parameters are read from environment variables"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'), \
             patch('retrieval.fusion.enforce_token_budget') as mock_budget:
            
            # Set up mocks
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.return_value = self.sample_faiss_docs
            
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.return_value = self.sample_neo4j_docs
            
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = mock_faiss_retriever
            mock_graph.return_value = mock_graph_retriever
            
            # Mock budget enforcement to return a subset
            mock_budget.return_value = self.sample_faiss_docs[:2]
            
            # Mock LLM
            mock_llm.return_value = self._create_mock_llm()
            
            # Initialize and test
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {"output_text": "Test response"}
            
            result = chatbot.chat("Test query")
            
            # Verify budget enforcement was called with environment parameters
            mock_budget.assert_called_once()
            call_args = mock_budget.call_args
            
            # Check budget parameter
            assert call_args.kwargs['budget'] == 2000  # From FUSION_TOKEN_BUDGET
            assert call_args.kwargs['diversity_factor'] == 0.7  # From FUSION_DIVERSITY_FACTOR
    
    def test_chat_history_context_in_dual_mode(self):
        """Test that chat history is properly handled in dual mode"""
        with patch('chatbot.FAISS') as mock_faiss, \
             patch('graph_rag.neo4j_client.GraphRetriever') as mock_graph, \
             patch.object(LLMProvider, 'get_llm') as mock_llm, \
             patch.object(LLMProvider, 'get_embeddings'):
            
            # Set up mocks
            mock_faiss_retriever = Mock()
            mock_faiss_retriever.get_relevant_documents.return_value = self.sample_faiss_docs
            
            mock_graph_retriever = Mock()
            mock_graph_retriever.get_relevant_documents.return_value = self.sample_neo4j_docs
            
            mock_vector_store = Mock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_vector_store.as_retriever.return_value = mock_faiss_retriever
            mock_graph.return_value = mock_graph_retriever
            
            # Mock LLM
            mock_llm.return_value = self._create_mock_llm()
            
            # Initialize chatbot
            chatbot = AstronomyChatbot(vector_store_path=self.temp_dir)
            chatbot.qa_chain = Mock()
            chatbot.qa_chain.invoke.return_value = {"output_text": "First response"}
            
            # First query
            result1 = chatbot.chat("What is dark energy?")
            assert len(chatbot.chat_history) == 1
            
            # Mock second response
            chatbot.qa_chain.invoke.return_value = {"output_text": "Follow-up response"}
            
            # Follow-up query
            result2 = chatbot.chat("How does it affect the universe?")
            assert len(chatbot.chat_history) == 2
            
            # Verify FAISS retriever was called with contextual query for follow-up
            faiss_calls = mock_faiss_retriever.get_relevant_documents.call_args_list
            assert len(faiss_calls) == 2
            
            # Second call should include context from first question
            second_call_query = faiss_calls[1][0][0]
            assert "What is dark energy?" in second_call_query  # Context from previous question
            assert "How does it affect the universe?" in second_call_query  # Current question


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
