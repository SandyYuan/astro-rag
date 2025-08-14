#!/usr/bin/env python3
"""
Manual testing script for dual retrieval mode.

This script allows you to quickly test the dual mode with real queries
and compare it against single modes.

Usage:
    python manual_test_dual_mode.py
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from chatbot import AstronomyChatbot

def test_mode(mode_name: str, queries: list) -> dict:
    """Test a specific retrieval mode with given queries"""
    print(f"\n{'='*60}")
    print(f"TESTING {mode_name.upper()} MODE")
    print(f"{'='*60}")
    
    # Set environment for this mode
    os.environ['RAG_MODE'] = mode_name
    
    try:
        # Initialize chatbot
        print(f"Initializing {mode_name} chatbot...")
        start_init = time.time()
        chatbot = AstronomyChatbot(vector_store_path="rag_data/index_all")
        init_time = time.time() - start_init
        print(f"✓ Initialized in {init_time:.2f}s")
        
        results = {}
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            start_time = time.time()
            result = chatbot.chat(query)
            duration = time.time() - start_time
            
            print(f"Response time: {duration:.2f}s")
            print(f"Sources ({len(result['sources'])}): {result['sources']}")
            print(f"Answer ({len(result['answer'])} chars):")
            print(f"  {result['answer'][:200]}...")
            
            results[query] = {
                'answer': result['answer'],
                'sources': result['sources'],
                'duration': duration
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to test {mode_name} mode: {e}")
        return {}

def compare_modes(queries: list):
    """Compare all three modes with the same queries"""
    modes = ['faiss', 'neo4j', 'dual']
    all_results = {}
    
    for mode in modes:
        all_results[mode] = test_mode(mode, queries)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        for mode in modes:
            if query in all_results[mode]:
                result = all_results[mode][query]
                print(f"{mode:>6}: {result['duration']:5.2f}s | {len(result['sources']):2d} sources | {len(result['answer']):4d} chars")
            else:
                print(f"{mode:>6}: FAILED")
    
    # Source diversity analysis
    print(f"\n{'='*30}")
    print("SOURCE DIVERSITY ANALYSIS")
    print(f"{'='*30}")
    
    for query in queries:
        print(f"\nQuery: {query[:40]}...")
        
        all_sources = set()
        mode_sources = {}
        
        for mode in modes:
            if query in all_results[mode]:
                sources = set(all_results[mode][query]['sources'])
                mode_sources[mode] = sources
                all_sources.update(sources)
        
        print(f"Total unique sources: {len(all_sources)}")
        for mode in modes:
            if mode in mode_sources:
                unique_to_mode = mode_sources[mode] - set().union(*(mode_sources[m] for m in modes if m != mode))
                print(f"  {mode:>6}: {len(mode_sources[mode]):2d} sources ({len(unique_to_mode)} unique)")

def main():
    """Main testing function"""
    print("🚀 Manual Dual Mode Testing")
    print("=" * 60)
    
    # Check prerequisites
    required_env = ['GOOGLE_API_KEY', 'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
    missing = [env for env in required_env if not os.getenv(env)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file or environment")
        return
    
    if not os.path.exists("rag_data/index_all"):
        print("❌ FAISS index not found at rag_data/index_all")
        print("Please run: python rag_processor.py")
        return
    
    print("✓ All prerequisites found")
    
    # Set fusion parameters for dual mode
    os.environ['FUSION_TOKEN_BUDGET'] = '3000'
    os.environ['FUSION_DIVERSITY_FACTOR'] = '0.7'
    
    # Test queries - mix of different types
    test_queries = [
        "What is the S8 tension?",
        "How do DES Y3 results compare to Planck?",
        "What constraints does weak lensing provide on dark matter?",
        "What are the main findings from galaxy clustering analysis?"
    ]
    
    print(f"Testing {len(test_queries)} queries across 3 modes...")
    
    try:
        compare_modes(test_queries)
        
        print(f"\n{'='*60}")
        print("✓ TESTING COMPLETE")
        print(f"{'='*60}")
        
        # Interactive mode
        print("\n🔍 Interactive Testing")
        print("Enter queries to test dual mode (or 'quit' to exit):")
        
        # Set to dual mode for interactive testing
        os.environ['RAG_MODE'] = 'dual'
        chatbot = AstronomyChatbot(vector_store_path="rag_data/index_all")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                start_time = time.time()
                result = chatbot.chat(query)
                duration = time.time() - start_time
                
                print(f"\nResponse ({duration:.2f}s):")
                print(f"Sources: {result['sources']}")
                print(f"Answer: {result['answer']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
