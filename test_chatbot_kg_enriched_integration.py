"""
Integration tests for KG-enriched pipeline in chatbot.

Tests the integration of the new sequential pipeline with the existing chatbot system.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from chatbot import AstronomyChatbot
from retrieval.kg_enriched_retrieval import KGEnrichedRetriever
from retrieval.kg_filter import KGQueryFilter


class TestChatbotKGEnrichedIntegration:
    """Test chatbot integration with KG-enriched pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'GOOGLE_API_KEY': 'test_key',
            'USE_KG_ENRICHED': 'true',
            'RAG_MODE': 'dual'  # Use dual mode as base
        })
        self.env_patcher.start()
        
        # Mock dependencies to avoid actual file/network operations
        self.mock_llm_provider = Mock()
        self.mock_graph_retriever = Mock()
        self.mock_vector_retriever = Mock()
        self.mock_kg_filter = Mock()
        
    def teardown_method(self):
        """Cleanup after tests."""
        self.env_patcher.stop()
    
    @patch('chatbot.LLMProvider')
    @patch('graph_rag.neo4j_client.GraphRetriever')
    @patch('chatbot.FAISS')
    @patch('retrieval.kg_filter.KGQueryFilter')
    @patch('retrieval.kg_enriched_retrieval.KGEnrichedRetriever')
    def test_kg_enriched_mode_initialization(
        self, 
        mock_kg_enriched_retriever_class,
        mock_kg_filter_class,
        mock_faiss,
        mock_graph_retriever_class,
        mock_llm_provider_class
    ):
        """Test chatbot initializes with KG-enriched mode when enabled."""
        # Setup mocks
        mock_llm_provider_class.return_value = self.mock_llm_provider
        mock_graph_retriever_class.return_value = self.mock_graph_retriever
        mock_kg_filter_class.return_value = self.mock_kg_filter
        mock_kg_enriched_retriever_class.return_value = Mock()
        
        mock_faiss.load_local.return_value = Mock()
        
        # Initialize chatbot
        with patch('os.path.exists', return_value=True):
            chatbot = AstronomyChatbot()
        
        # Verify KG-enriched mode is enabled
        assert hasattr(chatbot, 'use_kg_enriched_pipeline')
        assert chatbot.use_kg_enriched_pipeline is True
        
        # Verify KG-enriched retriever was created
        mock_kg_enriched_retriever_class.assert_called_once()
    
    @patch('chatbot.LLMProvider')
    @patch('graph_rag.neo4j_client.GraphRetriever')

    @patch('chatbot.FAISS')
    def test_kg_enriched_mode_disabled(
        self, 
        mock_faiss,
        mock_neo4j_client,
        mock_graph_retriever_class,
        mock_llm_provider_class
    ):
        """Test chatbot uses legacy mode when KG-enriched is disabled."""
        # Override environment to disable KG-enriched
        with patch.dict(os.environ, {'USE_KG_ENRICHED': 'false'}):
            mock_llm_provider_class.return_value = self.mock_llm_provider
            mock_graph_retriever_class.return_value = self.mock_graph_retriever
            mock_faiss.load_local.return_value = Mock()
            mock_neo4j_client.return_value = Mock()
            
            # Initialize chatbot
            with patch('os.path.exists', return_value=True):
                chatbot = AstronomyChatbot()
            
            # Verify KG-enriched mode is disabled
            assert hasattr(chatbot, 'use_kg_enriched_pipeline')
            assert chatbot.use_kg_enriched_pipeline is False
    
    @patch('chatbot.LLMProvider')
    @patch('graph_rag.neo4j_client.GraphRetriever')

    @patch('chatbot.FAISS')
    @patch('retrieval.kg_filter.KGQueryFilter')
    @patch('retrieval.kg_enriched_retrieval.KGEnrichedRetriever')
    def test_kg_enriched_retrieval_in_chat(
        self,
        mock_kg_enriched_retriever_class,
        mock_kg_filter_class,
        mock_faiss,
        mock_neo4j_client,
        mock_graph_retriever_class,
        mock_llm_provider_class
    ):
        """Test that KG-enriched retrieval is used during chat."""
        # Setup mocks
        mock_llm_provider_class.return_value = self.mock_llm_provider
        mock_graph_retriever_class.return_value = self.mock_graph_retriever
        mock_kg_filter_class.return_value = self.mock_kg_filter
        
        mock_kg_enriched_retriever = Mock()
        mock_kg_enriched_retriever_class.return_value = mock_kg_enriched_retriever
        
        # Mock retrieval results
        mock_documents = [
            Document(page_content="Dark matter detection content", metadata={"source": "test.pdf"})
        ]
        mock_kg_enriched_retriever.get_relevant_documents.return_value = mock_documents
        
        mock_faiss.load_local.return_value = Mock()
        
        # Mock LLM chain
        mock_qa_chain = Mock()
        mock_qa_chain.invoke.return_value = {"output_text": "Test response"}
        
        # Initialize chatbot
        with patch('os.path.exists', return_value=True):
            with patch('chatbot.load_qa_chain', return_value=mock_qa_chain):
                chatbot = AstronomyChatbot()
        
        # Test chat query
        query = "What is dark matter detection?"
        response = chatbot.chat(query)
        
        # Verify KG-enriched retriever was used
        mock_kg_enriched_retriever.get_relevant_documents.assert_called()
        
        # Verify response structure
        assert "answer" in response
        assert "sources" in response
        assert response["answer"] == "Test response"
    
    @patch('chatbot.LLMProvider')
    @patch('graph_rag.neo4j_client.GraphRetriever')  

    @patch('chatbot.FAISS')
    def test_backward_compatibility_with_existing_modes(
        self,
        mock_faiss,
        mock_neo4j_client,
        mock_graph_retriever_class,
        mock_llm_provider_class
    ):
        """Test backward compatibility with existing RAG_MODE options."""
        # Test with different RAG_MODE values
        for rag_mode in ['faiss', 'neo4j', 'dual']:
            with patch.dict(os.environ, {
                'RAG_MODE': rag_mode,
                'USE_KG_ENRICHED': 'false'  # Disable KG-enriched for this test
            }):
                mock_llm_provider_class.return_value = self.mock_llm_provider
                mock_graph_retriever_class.return_value = self.mock_graph_retriever
                mock_faiss.load_local.return_value = Mock()
                mock_neo4j_client.return_value = Mock()
                
                # Initialize chatbot - should not raise exception
                with patch('os.path.exists', return_value=True):
                    chatbot = AstronomyChatbot()
                
                # Verify mode is set correctly
                assert chatbot.rag_mode == rag_mode
                assert chatbot.use_kg_enriched_pipeline is False
    
    @patch('chatbot.LLMProvider')
    @patch('graph_rag.neo4j_client.GraphRetriever')
 
    @patch('chatbot.FAISS')
    @patch('retrieval.kg_filter.KGQueryFilter')
    @patch('retrieval.kg_enriched_retrieval.KGEnrichedRetriever')
    def test_agent_mode_compatibility(
        self,
        mock_kg_enriched_retriever_class,
        mock_kg_filter_class,
        mock_faiss,
        mock_neo4j_client,
        mock_graph_retriever_class,
        mock_llm_provider_class
    ):
        """Test KG-enriched mode works with agent mode."""
        # Setup environment for agent mode
        with patch.dict(os.environ, {
            'CHAT_MODE': 'agent',
            'USE_KG_ENRICHED': 'true',
            'RAG_MODE': 'dual'
        }):
            # Setup mocks
            mock_llm_provider_class.return_value = self.mock_llm_provider
            mock_graph_retriever_class.return_value = self.mock_graph_retriever
            mock_kg_filter_class.return_value = self.mock_kg_filter
            mock_kg_enriched_retriever_class.return_value = Mock()
            
            mock_faiss.load_local.return_value = Mock()
            mock_neo4j_client.return_value = Mock()
            
            # Mock agent setup
            mock_agent_executor = Mock()
            
            with patch('os.path.exists', return_value=True):
                with patch('chatbot.create_react_agent'):
                    with patch('chatbot.AgentExecutor', return_value=mock_agent_executor):
                        chatbot = AstronomyChatbot()
            
            # Verify both agent and KG-enriched modes are enabled
            assert chatbot.chat_mode == 'agent'
            assert chatbot.use_kg_enriched_pipeline is True
    
    @patch('chatbot.LLMProvider')
    @patch('graph_rag.neo4j_client.GraphRetriever')

    @patch('chatbot.FAISS')
    @patch('retrieval.kg_enriched_retrieval.KGEnrichedRetriever')
    def test_error_handling_fallback(
        self,
        mock_kg_enriched_retriever_class,
        mock_faiss,
        mock_neo4j_client,
        mock_graph_retriever_class,
        mock_llm_provider_class
    ):
        """Test error handling and fallback to legacy retrieval."""
        # Setup mocks
        mock_llm_provider_class.return_value = self.mock_llm_provider
        mock_graph_retriever_class.return_value = self.mock_graph_retriever
        
        # Mock KG-enriched retriever failure
        mock_kg_enriched_retriever = Mock()
        mock_kg_enriched_retriever.get_relevant_documents.side_effect = Exception("KG pipeline error")
        mock_kg_enriched_retriever_class.return_value = mock_kg_enriched_retriever
        
        mock_faiss.load_local.return_value = Mock()
        
        # Mock legacy retrieval success
        mock_fusion_retriever = Mock()
        mock_fusion_retriever.get_relevant_documents.return_value = [
            Document(page_content="Fallback content", metadata={"source": "fallback.pdf"})
        ]
        
        # Mock QA chain
        mock_qa_chain = Mock()
        mock_qa_chain.invoke.return_value = {"output_text": "Fallback response"}
        
        with patch('os.path.exists', return_value=True):
            with patch('chatbot.load_qa_chain', return_value=mock_qa_chain):
                with patch.object(AstronomyChatbot, '_dual_retrieval_with_fusion', return_value=mock_fusion_retriever.get_relevant_documents.return_value):
                    chatbot = AstronomyChatbot()
        
        # Test chat - should not raise exception
        response = chatbot.chat("What is dark matter?")
        
        # Verify fallback was used
        assert response["answer"] == "Fallback response"


if __name__ == "__main__":
    pytest.main([__file__])
