#!/usr/bin/env python3
"""
Comprehensive GraphRAG Test Suite

Tests all major improvements:
1. Coverage expansion (DES Y3 papers indexed)
2. Provenance enhancement (arXiv URLs as sources)
3. Relation structure (typed relationships)
4. End-to-end chatbot functionality
"""

import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Setup
load_dotenv()
sys.path.append('.')
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def test_neo4j_connection():
    """Test 1: Verify Neo4j connection and basic schema"""
    print("🔧 Test 1: Neo4j Connection & Schema")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        
        if not all([uri, user, password]):
            print("❌ Missing Neo4j environment variables")
            return False
            
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Test connection
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1
            
            # Check constraints exist
            constraints = session.run("SHOW CONSTRAINTS").data()
            constraint_names = {c.get('name', '') for c in constraints}
            required = {'entity_name', 'paper_path', 'claim_id'}
            
            if not required.issubset(constraint_names):
                print(f"❌ Missing constraints. Found: {constraint_names}")
                return False
                
            # Check indexes exist
            indexes = session.run("SHOW INDEXES").data()
            index_names = {idx.get('name', '') for idx in indexes}
            required_indexes = {'entityFulltext', 'claimFulltext', 'paperFulltext'}
            
            if not required_indexes.issubset(index_names):
                print(f"❌ Missing indexes. Found: {index_names}")
                return False
                
        driver.close()
        print("✅ Neo4j connection and schema verified")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_coverage_expansion():
    """Test 2: Verify DES Y3 papers are indexed with good coverage"""
    print("\n📊 Test 2: Coverage Expansion")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Count total papers
            result = session.run("MATCH (p:Paper) RETURN count(p) as total")
            total_papers = result.single()["total"]
            
            # Count DES Y3 papers specifically
            result = session.run(
                "MATCH (p:Paper) WHERE p.path CONTAINS 'Dark_Energy_Survey_Year_3' "
                "RETURN count(p) as des_y3_count"
            )
            des_y3_count = result.single()["des_y3_count"]
            
            # Check for S8-related entities
            result = session.run(
                "MATCH (e:Entity) WHERE e.name CONTAINS 'S8' OR e.name CONTAINS 'DES Y3' "
                "RETURN count(e) as s8_entities"
            )
            s8_entities = result.single()["s8_entities"]
            
            # Check for claims about S8
            result = session.run(
                "CALL db.index.fulltext.queryNodes('claimFulltext', 'S8') YIELD node "
                "RETURN count(node) as s8_claims"
            )
            s8_claims = result.single()["s8_claims"]
            
        driver.close()
        
        print(f"📈 Total papers indexed: {total_papers}")
        print(f"📈 DES Y3 papers: {des_y3_count}")
        print(f"📈 S8-related entities: {s8_entities}")
        print(f"📈 S8-related claims: {s8_claims}")
        
        # Validate coverage
        if total_papers < 15:
            print("❌ Insufficient paper coverage")
            return False
        if des_y3_count < 3:
            print("❌ Insufficient DES Y3 paper coverage")
            return False
        if s8_claims < 1:
            print("❌ No S8-related claims found")
            return False
            
        print("✅ Coverage expansion verified")
        return True
        
    except Exception as e:
        print(f"❌ Coverage test failed: {e}")
        return False

def test_provenance_enhancement():
    """Test 3: Verify arXiv URLs are used as sources"""
    print("\n🔗 Test 3: Provenance Enhancement (arXiv URLs)")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Check papers with arXiv URLs
            result = session.run(
                "MATCH (p:Paper) WHERE p.arxiv_url IS NOT NULL "
                "RETURN count(p) as arxiv_papers, collect(p.arxiv_url)[..3] as sample_urls"
            )
            record = result.single()
            arxiv_papers = record["arxiv_papers"]
            sample_urls = record["sample_urls"]
            
            print(f"📄 Papers with arXiv URLs: {arxiv_papers}")
            print(f"📄 Sample URLs: {sample_urls}")
            
            if arxiv_papers < 3:
                print("❌ Insufficient papers with arXiv URLs")
                return False
                
            # Validate URL format
            for url in sample_urls:
                if not url.startswith('http://arxiv.org/'):
                    print(f"❌ Invalid arXiv URL format: {url}")
                    return False
                    
        driver.close()
        
        # Test retriever uses arXiv URLs
        from graph_rag.neo4j_client import GraphRetriever
        
        retriever = GraphRetriever(k=3)
        docs = retriever.get_relevant_documents("ADDGALS simulated catalogs")
        retriever.close()
        
        arxiv_sources = [d.metadata.get('source', '') for d in docs if d.metadata.get('source', '').startswith('http://arxiv.org/')]
        
        print(f"🔍 Retriever arXiv sources: {len(arxiv_sources)}")
        
        if len(arxiv_sources) < 1:
            print("❌ Retriever not returning arXiv URLs as sources")
            return False
            
        print("✅ Provenance enhancement verified")
        return True
        
    except Exception as e:
        print(f"❌ Provenance test failed: {e}")
        return False

def test_relation_structure():
    """Test 4: Verify typed relationships exist"""
    print("\n🔗 Test 4: Relation Structure (Typed Relationships)")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Get all relationship types
            result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type ORDER BY rel_type")
            rel_types = [record["rel_type"] for record in result]
            
            print(f"🔗 Relationship types: {rel_types}")
            
            # Check for semantic relationship types
            semantic_types = {'MEASURES', 'PREDICTS', 'USES', 'CONSTRAINS', 'SUPPORTS'}
            found_semantic = set(rel_types) & semantic_types
            
            print(f"🔗 Semantic relationship types found: {found_semantic}")
            
            # Count typed relationships
            typed_count = 0
            for rel_type in found_semantic:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count = result.single()["count"]
                typed_count += count
                print(f"   {rel_type}: {count} relationships")
                
        driver.close()
        
        if len(found_semantic) < 1:
            print("❌ No semantic relationship types found")
            return False
            
        if typed_count < 1:
            print("❌ No typed relationships found")
            return False
            
        print("✅ Relation structure verified")
        return True
        
    except Exception as e:
        print(f"❌ Relation structure test failed: {e}")
        return False

def test_graphrag_retriever():
    """Test 5: GraphRAG retriever functionality"""
    print("\n🔍 Test 5: GraphRAG Retriever")
    
    try:
        from graph_rag.neo4j_client import GraphRetriever
        
        retriever = GraphRetriever(k=5)
        
        # Test entity-centric search
        docs = retriever.get_relevant_documents("DES Y3 S8 measurements")
        
        print(f"🔍 Retrieved {len(docs)} documents")
        
        if len(docs) == 0:
            print("❌ No documents retrieved")
            retriever.close()
            return False
            
        # Check document structure
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            entity = doc.metadata.get('entity', 'unknown')
            content_preview = doc.page_content[:100].replace('\n', ' ')
            
            print(f"   Doc {i+1}: {entity} | Source: {source}")
            print(f"           Content: {content_preview}...")
            
        # Validate sources include arXiv URLs or entity names
        sources = [doc.metadata.get('source', '') for doc in docs]
        has_arxiv = any(s.startswith('http://arxiv.org/') for s in sources)
        has_entities = any(not s.startswith('http://') for s in sources)
        
        if not (has_arxiv or has_entities):
            print("❌ Invalid source formats")
            retriever.close()
            return False
            
        retriever.close()
        print("✅ GraphRAG retriever verified")
        return True
        
    except Exception as e:
        print(f"❌ GraphRAG retriever test failed: {e}")
        return False

def test_chatbot_integration():
    """Test 6: End-to-end chatbot with Neo4j mode"""
    print("\n🤖 Test 6: Chatbot Integration (Neo4j Mode)")
    
    try:
        from chatbot import AstronomyChatbot
        
        # Test Neo4j mode
        os.environ['RAG_MODE'] = 'neo4j'
        chatbot = AstronomyChatbot(vector_store_path='rag_data/index_all')
        chatbot.qa_chain.verbose = False
        
        # Test scientific parameter query
        response = chatbot.chat("What do DES Y3 results say about S8?")
        
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        
        print(f"🤖 Answer length: {len(answer)} characters")
        print(f"🤖 Number of sources: {len(sources)}")
        print(f"🤖 Answer preview: {answer[:200]}...")
        print(f"🤖 Sources: {sources}")
        
        # Validate response quality
        if len(answer) < 100:
            print("❌ Answer too short")
            return False
            
        if len(sources) == 0:
            print("❌ No sources provided")
            return False
            
        # Check for scientific content
        scientific_terms = ['S8', 'DES', 'cosmolog', 'survey', 'parameter']
        has_scientific_content = any(term.lower() in answer.lower() for term in scientific_terms)
        
        if not has_scientific_content:
            print("❌ Answer lacks scientific content")
            return False
            
        # Check source quality (should include arXiv URLs or reasonable entity names)
        good_sources = 0
        for source in sources:
            if source.startswith('http://arxiv.org/') or any(term in source for term in ['DES', 'Survey', 'Planck']):
                good_sources += 1
                
        if good_sources == 0:
            print("❌ No high-quality sources")
            return False
            
        print("✅ Chatbot integration verified")
        return True
        
    except Exception as e:
        print(f"❌ Chatbot integration test failed: {e}")
        return False

def test_faiss_comparison():
    """Test 7: Compare Neo4j vs FAISS modes"""
    print("\n⚖️  Test 7: Neo4j vs FAISS Comparison")
    
    try:
        from chatbot import AstronomyChatbot
        
        query = "Tell me about S8 tension in cosmology"
        
        # Test Neo4j mode
        os.environ['RAG_MODE'] = 'neo4j'
        chatbot_neo4j = AstronomyChatbot(vector_store_path='rag_data/index_all')
        chatbot_neo4j.qa_chain.verbose = False
        response_neo4j = chatbot_neo4j.chat(query)
        
        # Test FAISS mode  
        os.environ['RAG_MODE'] = 'faiss'
        chatbot_faiss = AstronomyChatbot(vector_store_path='rag_data/index_all')
        chatbot_faiss.qa_chain.verbose = False
        response_faiss = chatbot_faiss.chat(query)
        
        # Compare responses
        neo4j_answer = response_neo4j.get('answer', '')
        neo4j_sources = response_neo4j.get('sources', [])
        
        faiss_answer = response_faiss.get('answer', '')
        faiss_sources = response_faiss.get('sources', [])
        
        print(f"📊 Neo4j: {len(neo4j_answer)} chars, {len(neo4j_sources)} sources")
        print(f"📊 FAISS:  {len(faiss_answer)} chars, {len(faiss_sources)} sources")
        
        print(f"📊 Neo4j sources: {neo4j_sources[:3]}")
        print(f"📊 FAISS sources: {faiss_sources[:3]}")
        
        # Both should work
        if len(neo4j_answer) < 50 or len(faiss_answer) < 50:
            print("❌ One or both modes produced insufficient answers")
            return False
            
        if len(neo4j_sources) == 0 or len(faiss_sources) == 0:
            print("❌ One or both modes provided no sources")
            return False
            
        print("✅ Both Neo4j and FAISS modes working")
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("🧪 GraphRAG Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Neo4j Connection & Schema", test_neo4j_connection),
        ("Coverage Expansion", test_coverage_expansion),
        ("Provenance Enhancement", test_provenance_enhancement),
        ("Relation Structure", test_relation_structure),
        ("GraphRAG Retriever", test_graphrag_retriever),
        ("Chatbot Integration", test_chatbot_integration),
        ("Neo4j vs FAISS Comparison", test_faiss_comparison),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏆 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! GraphRAG system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
