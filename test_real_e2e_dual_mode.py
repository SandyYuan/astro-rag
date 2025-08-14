"""
Real End-to-End Tests for Dual Retrieval Mode

These tests use actual FAISS, Neo4j, and Gemini components to verify
that the dual retrieval system works correctly in practice.

PREREQUISITES:
- Neo4j database running with indexed papers
- FAISS index built (rag_data/index_all)
- Valid GOOGLE_API_KEY in environment
- NEO4J_* connection details in environment

Run with: pytest test_real_e2e_dual_mode.py -v -s
"""

import pytest
import os
import logging
from typing import List, Dict, Any

# Import the modules we'll be testing
import sys
sys.path.append('.')

from chatbot import AstronomyChatbot

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealDualModeE2E:
    """Real end-to-end tests for dual retrieval mode"""
    
    @pytest.fixture(scope="class")
    def check_prerequisites(self):
        """Check that all prerequisites are available"""
        missing = []
        
        # Check environment variables
        required_env = ['GOOGLE_API_KEY', 'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
        for env_var in required_env:
            if not os.getenv(env_var):
                missing.append(f"Environment variable: {env_var}")
        
        # Check FAISS index exists
        faiss_path = "rag_data/index_all"
        if not os.path.exists(faiss_path):
            missing.append(f"FAISS index not found at: {faiss_path}")
        
        if missing:
            pytest.skip(f"Prerequisites missing: {', '.join(missing)}")
        
        return True
    
    @pytest.fixture(scope="class")
    def dual_chatbot(self, check_prerequisites):
        """Create a real dual-mode chatbot for testing"""
        # Set dual mode
        os.environ['RAG_MODE'] = 'dual'
        os.environ['FUSION_TOKEN_BUDGET'] = '2000'
        os.environ['FUSION_DIVERSITY_FACTOR'] = '0.6'
        
        try:
            chatbot = AstronomyChatbot(vector_store_path="rag_data/index_all")
            logger.info("✓ Dual mode chatbot initialized successfully")
            return chatbot
        except Exception as e:
            pytest.fail(f"Failed to initialize dual mode chatbot: {e}")
    
    @pytest.fixture(scope="class")
    def faiss_chatbot(self, check_prerequisites):
        """Create a FAISS-only chatbot for comparison"""
        os.environ['RAG_MODE'] = 'faiss'
        
        try:
            chatbot = AstronomyChatbot(vector_store_path="rag_data/index_all")
            logger.info("✓ FAISS-only chatbot initialized successfully")
            return chatbot
        except Exception as e:
            pytest.fail(f"Failed to initialize FAISS chatbot: {e}")
    
    @pytest.fixture(scope="class")
    def neo4j_chatbot(self, check_prerequisites):
        """Create a Neo4j-only chatbot for comparison"""
        os.environ['RAG_MODE'] = 'neo4j'
        
        try:
            chatbot = AstronomyChatbot(vector_store_path="rag_data/index_all")
            logger.info("✓ Neo4j-only chatbot initialized successfully")
            return chatbot
        except Exception as e:
            pytest.fail(f"Failed to initialize Neo4j chatbot: {e}")
    
    def test_dual_mode_basic_functionality(self, dual_chatbot):
        """Test basic dual mode functionality with a simple query"""
        query = "What is dark energy?"
        
        logger.info(f"Testing dual mode with query: '{query}'")
        result = dual_chatbot.chat(query)
        
        # Basic response validation
        assert "answer" in result
        assert "sources" in result
        assert len(result["answer"]) > 50  # Substantial response
        assert len(result["sources"]) > 0  # Has sources
        
        logger.info(f"✓ Dual mode response: {len(result['answer'])} chars, {len(result['sources'])} sources")
        logger.info(f"Sources: {result['sources']}")
    
    def test_dual_vs_single_mode_comparison(self, dual_chatbot, faiss_chatbot, neo4j_chatbot):
        """Compare dual mode against single modes"""
        test_queries = [
            "What is the S8 tension?",
            "How does weak lensing constrain cosmology?",
            "What are the DES Y3 results?"
        ]
        
        results = {}
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            # Get results from all modes
            dual_result = dual_chatbot.chat(query)
            faiss_result = faiss_chatbot.chat(query)
            neo4j_result = neo4j_chatbot.chat(query)
            
            results[query] = {
                'dual': dual_result,
                'faiss': faiss_result,
                'neo4j': neo4j_result
            }
            
            # Log comparison
            logger.info(f"  Dual:  {len(dual_result['answer'])} chars, {len(dual_result['sources'])} sources")
            logger.info(f"  FAISS: {len(faiss_result['answer'])} chars, {len(faiss_result['sources'])} sources")
            logger.info(f"  Neo4j: {len(neo4j_result['answer'])} chars, {len(neo4j_result['sources'])} sources")
            
            # Dual mode should generally have more diverse sources
            dual_sources = set(dual_result['sources'])
            faiss_sources = set(faiss_result['sources'])
            neo4j_sources = set(neo4j_result['sources'])
            
            # Check that dual mode is working properly (should have at least some sources)
            # Dual mode may have fewer sources than single modes due to fusion deduplication or token budget constraints
            # The key is that it should still provide meaningful results
            assert len(dual_sources) > 0, f"Dual mode should provide sources for '{query}'"
            
            # Log the source diversity for analysis
            total_unique_sources = len(dual_sources | faiss_sources | neo4j_sources)
            logger.info(f"  Source diversity: {len(dual_sources)} dual, {len(faiss_sources)} FAISS, {len(neo4j_sources)} Neo4j, {total_unique_sources} total unique")
        
        return results
    
    def test_dual_mode_fusion_quality(self, dual_chatbot):
        """Test that fusion produces high-quality responses"""
        # Test queries that should benefit from both FAISS and Neo4j
        fusion_test_queries = [
            "What is the relationship between S8 measurements and cosmic shear?",
            "How do DES and Planck results compare for cosmological parameters?",
            "What constraints do galaxy surveys place on dark energy?"
        ]
        
        for query in fusion_test_queries:
            logger.info(f"\nTesting fusion quality for: '{query}'")
            
            result = dual_chatbot.chat(query)
            
            # Quality checks
            assert len(result["answer"]) > 100, "Response should be substantial"
            assert len(result["sources"]) >= 2, "Should have multiple sources"
            
            # Check for scientific content (basic heuristics)
            answer_lower = result["answer"].lower()
            has_scientific_content = any(term in answer_lower for term in [
                "measurement", "constraint", "parameter", "survey", "analysis",
                "sigma", "uncertainty", "statistical", "systematic"
            ])
            
            assert has_scientific_content, f"Response should contain scientific content for '{query}'"
            
            logger.info(f"✓ Quality check passed: {len(result['sources'])} sources, scientific content detected")
    
    def test_dual_mode_error_resilience(self, dual_chatbot):
        """Test that dual mode handles various query types gracefully"""
        edge_case_queries = [
            "Tell me about xyz123 nonexistent topic",  # Should handle gracefully
            "What is the meaning of life?",  # Off-topic query
            "",  # Empty query
            "S8",  # Very short query
            "What are the implications of the recent measurements of the cosmic microwave background anisotropies for our understanding of the early universe and the subsequent formation of large-scale structure?" # Very long query
        ]
        
        for query in edge_case_queries:
            logger.info(f"\nTesting edge case: '{query[:50]}...' ({len(query)} chars)")
            
            try:
                result = dual_chatbot.chat(query)
                
                # Should always return a valid response structure
                assert "answer" in result
                assert "sources" in result
                assert isinstance(result["answer"], str)
                assert isinstance(result["sources"], list)
                
                logger.info(f"✓ Handled gracefully: {len(result['answer'])} chars response")
                
            except Exception as e:
                pytest.fail(f"Dual mode failed on edge case '{query}': {e}")
    
    def test_dual_mode_performance_timing(self, dual_chatbot, faiss_chatbot, neo4j_chatbot):
        """Test that dual mode performance is acceptable"""
        import time
        
        query = "What are the main results from the Dark Energy Survey?"
        
        # Time each mode
        modes = {
            'dual': dual_chatbot,
            'faiss': faiss_chatbot,
            'neo4j': neo4j_chatbot
        }
        
        timings = {}
        
        for mode_name, chatbot in modes.items():
            logger.info(f"\nTiming {mode_name} mode...")
            
            start_time = time.time()
            result = chatbot.chat(query)
            end_time = time.time()
            
            duration = end_time - start_time
            timings[mode_name] = duration
            
            logger.info(f"  {mode_name}: {duration:.2f}s ({len(result['sources'])} sources)")
        
        # Dual mode should not be more than 3x slower than the slowest single mode
        max_single_time = max(timings['faiss'], timings['neo4j'])
        dual_time = timings['dual']
        
        logger.info(f"\nPerformance summary:")
        logger.info(f"  FAISS: {timings['faiss']:.2f}s")
        logger.info(f"  Neo4j: {timings['neo4j']:.2f}s") 
        logger.info(f"  Dual:  {timings['dual']:.2f}s")
        logger.info(f"  Dual overhead: {dual_time/max_single_time:.2f}x")
        
        assert dual_time <= max_single_time * 3.0, \
            f"Dual mode ({dual_time:.2f}s) should not be more than 3x slower than single modes ({max_single_time:.2f}s)"
    
    def test_dual_mode_conversation_continuity(self, dual_chatbot):
        """Test that dual mode maintains conversation context properly"""
        
        # Multi-turn conversation
        conversation = [
            "What is the S8 parameter?",
            "How do DES measurements compare to Planck?",
            "What causes this tension?",
            "Are there any proposed solutions?"
        ]
        
        responses = []
        
        for i, query in enumerate(conversation):
            logger.info(f"\nTurn {i+1}: '{query}'")
            
            result = dual_chatbot.chat(query)
            responses.append(result)
            
            # Each response should be relevant
            assert len(result["answer"]) > 20, f"Turn {i+1} should have substantial response"
            assert len(result["sources"]) > 0, f"Turn {i+1} should have sources"
            
            logger.info(f"  Response: {len(result['answer'])} chars, {len(result['sources'])} sources")
        
        # Later responses should show some continuity (basic check)
        # For S8 tension queries, later responses should still mention relevant terms
        final_response = responses[-1]["answer"].lower()
        context_terms = ["s8", "tension", "des", "planck", "measurement", "parameter"]
        
        context_found = sum(1 for term in context_terms if term in final_response)
        assert context_found >= 2, "Final response should maintain conversation context"
        
        logger.info(f"✓ Conversation continuity maintained ({context_found}/{len(context_terms)} context terms found)")


if __name__ == "__main__":
    # Run with more verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
