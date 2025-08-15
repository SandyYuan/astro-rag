# KG-Enriched Vector Search Implementation

## Overview
Implement a new retrieval pipeline that uses Knowledge Graph results to enrich vector search queries, replacing the current parallel fusion approach with a sequential KG-guided approach.

## Current vs Proposed Architecture

**Current:** `User Query → [KG ∥ Vector] → Fusion → Results`  
**Proposed:** `User Query → KG → LLM Filter → Query Enrichment → Vector Search → Results`

---

## Task 1: Create Test Suite for KG-Enriched Pipeline

**File:** `test_kg_enriched_pipeline.py`

### 1.1 Test Data Setup
- [ ] Create mock KG results with relevant/irrelevant content
- [ ] Create test queries covering different astronomy topics
- [ ] Mock Gemini Flash LLM responses for consistent testing

### 1.2 Core Pipeline Tests
```python
def test_kg_enriched_pipeline_basic_flow():
    """Test complete pipeline: query → KG → LLM filter → vector search"""
    
def test_kg_enriched_pipeline_with_empty_kg():
    """Test fallback when KG returns no results"""
    
def test_kg_enriched_pipeline_with_llm_failure():
    """Test fallback when LLM filtering fails"""
```

### 1.3 LLM Filtering Tests
```python
def test_llm_filter_removes_irrelevant_content():
    """Test that cosmic inflation gets filtered out from dark matter detection query"""
    
def test_llm_filter_preserves_relevant_content():
    """Test that WIMP detection content is preserved for dark matter query"""
    
def test_llm_filter_formats_for_vector_search():
    """Test output format is optimized for vector search"""
```

### 1.4 Query Enrichment Tests
```python
def test_query_enrichment_with_kg_context():
    """Test enriched query contains both original query and KG context"""
    
def test_query_enrichment_fallback():
    """Test original query is used when no KG context available"""
```

---

## Task 2: Implement LLM-Based KG Filtering Component

**File:** `retrieval/kg_filter.py`

### 2.1 Core Filtering Class
```python
class KGQueryFilter:
    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider for Gemini Flash calls"""
        
    def filter_and_format_kg_results(self, kg_results: List[Dict], user_query: str) -> str:
        """Main method: filter KG results and format for vector search"""
        
    def _format_kg_results_for_llm(self, kg_results: List[Dict]) -> str:
        """Format KG results into structured LLM input"""
        
    def _create_filtering_prompt(self, user_query: str, kg_content: str) -> str:
        """Create optimized prompt for Gemini Flash filtering"""
        
    def _fallback_formatting(self, kg_results: List[Dict], user_query: str) -> str:
        """Simple rule-based fallback if LLM fails"""
```

### 2.2 Write Tests First (TDD)
- [ ] Test LLM prompt generation
- [ ] Test successful filtering and formatting
- [ ] Test fallback behavior when LLM fails
- [ ] Test input validation and edge cases

### 2.3 Implementation Requirements
- [ ] Use Gemini 2.5 Flash for speed/cost optimization
- [ ] Limit KG input to prevent token overflow (max 12-15 results)
- [ ] Temperature=0.0 for deterministic results
- [ ] Robust error handling with fallback
- [ ] Structured logging for debugging

---

## Task 3: Implement KG-Enriched Retrieval Pipeline

**File:** `retrieval/kg_enriched_retrieval.py`

### 3.1 Pipeline Class
```python
class KGEnrichedRetriever:
    def __init__(self, graph_retriever, vector_retriever, kg_filter: KGQueryFilter):
        """Initialize with existing retrievers and new KG filter"""
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Main pipeline method"""
        
    def _convert_kg_documents_to_dict(self, kg_docs: List[Document]) -> List[Document]:
        """Convert LangChain Documents to dict format for LLM processing"""
        
    def _create_enriched_query(self, original_query: str, kg_context: str) -> str:
        """Combine original query with KG context"""
```

### 3.2 Write Tests First (TDD)
- [ ] Test complete pipeline with mock components
- [ ] Test document format conversion
- [ ] Test query enrichment logic
- [ ] Test error handling at each step
- [ ] Test performance characteristics (timing)

### 3.3 Implementation Requirements
- [ ] Comprehensive logging for each pipeline step
- [ ] Graceful degradation when components fail
- [ ] Preserve original query intent in enriched query
- [ ] Return standard LangChain Document format

---

## Task 4: Integration with Existing Chatbot

**File:** `chatbot.py`

### 4.1 Add KG-Enriched Mode
```python
# Add to chatbot.__init__()
self.use_kg_enriched_pipeline = os.environ.get("USE_KG_ENRICHED", "false").lower() == "true"

# Update _dual_retrieval_with_fusion method
def _dual_retrieval_with_fusion(self, query: str, standalone_question: str) -> List[Document]:
    if self.use_kg_enriched_pipeline:
        return self._kg_enriched_retrieval(standalone_question)
    else:
        return self._current_parallel_fusion(query, standalone_question)
```

### 4.2 Write Integration Tests First (TDD)
- [ ] Test mode toggle functionality
- [ ] Test integration with existing RAG_MODE options
- [ ] Test backward compatibility with current system
- [ ] Test agent mode compatibility

### 4.3 Implementation Requirements
- [ ] Environment variable toggle: `USE_KG_ENRICHED=true/false`
- [ ] Maintain all existing functionality as fallback
- [ ] Update logging to indicate which pipeline is used
- [ ] No breaking changes to existing API

---

## Task 5: Performance Testing & Validation

**File:** `test_kg_enriched_performance.py`

### 5.1 Performance Tests
```python
def test_pipeline_latency():
    """Measure end-to-end latency vs current approach"""
    
def test_llm_call_timing():
    """Measure LLM filtering step specifically"""
    
def test_memory_usage():
    """Ensure no memory leaks in pipeline"""
```

### 5.2 Quality Tests
```python
def test_retrieval_quality_improvement():
    """Compare retrieval relevance: new vs old approach"""
    
def test_scientific_concept_coverage():
    """Verify KG enrichment improves domain-specific retrieval"""
    
def test_edge_case_handling():
    """Test with novel queries, empty results, etc."""
```

### 5.3 A/B Testing Framework
- [ ] Create test queries covering different astronomy topics
- [ ] Implement side-by-side comparison function
- [ ] Measure relevance scores for retrieved documents
- [ ] Generate performance comparison report

---

## Task 6: Documentation & Deployment

### 6.1 Update Configuration
- [ ] Add `USE_KG_ENRICHED=true` to environment variables documentation
- [ ] Update `status.md` with new pipeline description
- [ ] Add troubleshooting guide for new pipeline

### 6.2 Deployment Preparation
- [ ] Ensure Gemini Flash API access is configured
- [ ] Test with real Neo4j database and FAISS index
- [ ] Validate with actual astronomy research queries
- [ ] Create rollback plan (set `USE_KG_ENRICHED=false`)

---

## Acceptance Criteria

### Functional Requirements
- [ ] New pipeline produces relevant results for astronomy queries
- [ ] Fallback to original query works when KG/LLM fails
- [ ] Performance overhead < 500ms per query
- [ ] Backward compatibility maintained (existing functionality unchanged)

### Quality Requirements
- [ ] All tests pass with >90% coverage
- [ ] No regression in current system performance
- [ ] Improved relevance for broad scientific queries
- [ ] Robust error handling and logging

### Deployment Requirements
- [ ] Feature flag allows easy toggle between approaches
- [ ] Clear documentation for configuration and troubleshooting
- [ ] Performance monitoring and alerting ready

---

## Implementation Order

1. **Task 1** - Test suite (establishes requirements)
2. **Task 2** - KG filtering component (core new functionality)
3. **Task 3** - Pipeline implementation (orchestrates components)
4. **Task 4** - Chatbot integration (connects to existing system)
5. **Task 5** - Performance validation (ensures quality)
6. **Task 6** - Documentation and deployment (production readiness)

**Estimated Timeline:** 2-3 days for experienced engineer following TDD approach.
