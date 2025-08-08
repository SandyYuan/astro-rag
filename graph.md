## Neo4j GraphRAG Integration Plan

Objective: Add a Neo4j-based GraphRAG path alongside the existing FAISS RAG with minimal edits. Keep the FastAPI surface (`app.py`) and chat flow intact. Allow easy A/B testing by switching retrievers.

### 1) Architecture delta (what changes vs what stays)
- Keep
  - PDF collection (`paper_collector.py`) and current FAISS pipeline (`rag_processor.py`)
  - Web/API surface in `app.py`, persona/system prompt and QA chain orchestration in `chatbot.py`
- Add
  - A small `graph_rag/` module containing:
    - `graph_rag/index.py`: one-off (or scheduled) graph indexing from existing `.txt`/PDF-derived text
    - `graph_rag/neo4j_client.py`: a `GraphRetriever` that returns `langchain`-compatible `Document` objects
  - A mode switch in `chatbot.py` to select `neo4j` vs `faiss` retriever (default remains FAISS)

Result: Zero API changes, minimal code edits localized to `chatbot.py` plus new isolated files.

### 2) Prerequisites
- Neo4j
  - Option A: Neo4j AuraDB (recommended for simplicity)
  - Option B: Local Docker
    - Example: `docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5`
- Python deps (add to `requirements.txt` or a separate `requirements-graph.txt`):
  - `neo4j` (official Python driver)
  - A GraphRAG helper library if you choose to leverage one (see official Neo4j GraphRAG docs)
  - Your LLM/embedding provider SDK (OpenAI/Azure recommended for graph extraction; Gemini can remain for answer generation)

Environment variables (in `.env`):
```
NEO4J_URI=bolt://localhost:7687         # or Aura bolt URI
NEO4J_USER=neo4j
NEO4J_PASSWORD=********
# For extraction/embedding during graph build (choose one stack):
OPENAI_API_KEY=...
# or AZURE_OPENAI_* vars if using Azure
```

### 3) Data sources
- Use existing text corpus produced by your repo:
  - `papers/` and `papers_np/` contain `.txt` metadata/abstracts; PDFs in `papers/` can be converted if needed
- Optional: create `rag_graph/input/` and copy curated `.txt` files for iterative indexing on subsets first

### 4) Graph schema (minimal, practical)
- Nodes
  - `Paper {id, title, path, year, url}`
  - `Entity {name, type}` (type optional; e.g., Person, Object, Method, Dataset)
  - `Claim {id, text, confidence}` (optional summarization granularity)
- Relationships
  - `(Entity)-[:MENTIONED_IN]->(Paper) {spans: [start,end], freq}` (optional props)
  - `(Entity)-[:RELATES_TO]->(Entity) {relation_type, evidence}`
  - `(Claim)-[:SUPPORTED_BY]->(Paper)`
  - Optionally `(Claim)-[:ABOUT]->(Entity)`

Indexes/constraints (run once):
```cypher
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name) IS UNIQUE;
CREATE CONSTRAINT paper_path  IF NOT EXISTS FOR (p:Paper)  REQUIRE (p.path) IS UNIQUE;
CREATE CONSTRAINT claim_id    IF NOT EXISTS FOR (c:Claim)  REQUIRE (c.id) IS UNIQUE;
```

### 5) Indexing pipeline (new: `graph_rag/index.py`)
Goal: Extract entities/relations/claims from text and upsert into Neo4j. Keep it deterministic and restartable.

Steps
1. Load text files from `papers/` and `papers_np/` (start with a subset)
2. Chunk text (re-use `RecursiveCharacterTextSplitter` or simpler fixed-size chunks)
3. For each chunk:
   - Use your chosen LLM to extract:
     - normalized entity names and types
     - typed relations between entities (relation_type + minimal evidence span)
     - optional claims (short declarative statements)
   - Upsert nodes and relationships via Cypher using Neo4j driver
4. Optionally compute and store:
   - node-level/community summaries
   - entity/claim embeddings and create a Neo4j vector index for hybrid search

Operational notes
- Batch writes (transaction per document or per N chunks)
- Idempotent upserts (MERGE with deterministic keys: entity by `name`, paper by `path`, claim by `id`)
- Log progress and failures; support resume by tracking processed files in a small checkpoint file

### 6) Graph retriever (new: `graph_rag/neo4j_client.py`)
Purpose: Provide `get_relevant_documents(query)` returning `List[Document]` compatible with current `chatbot.py`.

Suggested behavior
- Detect query intent (broad vs entity-specific) with a light heuristic or a small LLM call
- Global query path: retrieve top community summaries / highest-degree or central entities and their key claims
- Local query path: resolve candidate entities; fetch the k-hop neighborhood and top claims with supporting papers
- Build a compact textual context block per result (include provenance)
- Return `langchain.schema.Document` instances with:
  - `page_content`: synthesized context (entity/claim snippets with pointers)
  - `metadata`: `{source: <paper_path or entity_name>, score: <optional>}` so UI can still show sources

Cypher patterns (illustrative)
```cypher
// Entities related to a phrase
CALL db.index.fulltext.queryNodes('entityNameIndex', $phrase) YIELD node, score
RETURN node, score LIMIT 10;

// Neighborhood claims
MATCH (e:Entity {name: $name})-[:RELATES_TO]-(e2)-[:ABOUT]-(c:Claim)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(p:Paper)
RETURN c.text AS claim, collect(DISTINCT p.path)[..3] AS sources LIMIT 20;
```

### 7) Wire-up in `chatbot.py` (minimal edit)
- Add a constructor parameter (or env var) to choose retrieval mode:
  - `retrieval_mode = os.getenv("RAG_MODE", "faiss")`
- In `setup_rag()`:
  - If `faiss`: keep existing code
  - If `neo4j`: instantiate `GraphRetriever` and assign to `self.retriever`; keep the same LLM and QA chain
- No other changes to `chat()` are needed since it already calls `self.retriever.get_relevant_documents(...)`

Example toggle (conceptual)
```python
# chatbot.py
self.retrieval_mode = os.getenv("RAG_MODE", "faiss")
if self.retrieval_mode == "neo4j":
    from graph_rag.neo4j_client import GraphRetriever
    self.retriever = GraphRetriever(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    )
else:
    # existing FAISS setup
```

### 8) Runbook
1. Start Neo4j
   - Aura: create DB, capture URI/user/password
   - Local Docker: run container and ensure Bolt on `7687`
2. Set environment
   - Update `.env` with `NEO4J_*` and LLM keys
3. Initialize graph
   - Run schema Cypher constraints (once)
4. Index a small subset first
   - `python -m graph_rag.index --limit 20` (implement `--limit` in the script for quick iteration)
5. Launch app with Neo4j mode
   - `RAG_MODE=neo4j python app.py`
6. A/B test
   - Switch `RAG_MODE=faiss` vs `neo4j` and compare multi-hop/entity questions and source quality

### 9) A/B evaluation (lightweight)
- Create a CSV of 15–20 representative questions (broad + entity-specific)
- For each mode, log: answer, latency, sources used
- Manually rate answer quality and source faithfulness
- If GraphRAG outperforms on target queries, scale indexing to full corpus

### 10) Performance, cost, and reliability
- Indexing is the expensive step; batch and cache extraction results
- Start with low temperature for extraction; prefer deterministic prompts/templates
- Add retries/backoff on LLM calls; checkpoint processed files
- Consider Neo4j vector indexes for hybrid retrieval once correctness is validated

### 11) Security and ops
- Store secrets only in `.env`; do not commit
- Restrict Neo4j to local network or use Aura with role-based access
- Add simple health checks: can connect to Neo4j; basic Cypher query returns

### 12) Rollback
- Set `RAG_MODE=faiss` and restart; no data/model changes needed

### 13) Deliverables checklist
- [ ] `graph_rag/index.py` (ingestion + extraction + upsert)
- [ ] `graph_rag/neo4j_client.py` (`GraphRetriever.get_relevant_documents`)
- [ ] `chatbot.py` mode switch in `setup_rag()`
- [ ] `.env` updated with `NEO4J_*` and LLM keys
- [ ] Constraints created in Neo4j
- [ ] Smoke test queries pass in both modes

### 14) Time estimate
- Neo4j setup + schema: 0.5h–1h
- Indexer (subset) + retriever scaffolding: 3h–5h
- Wiring + A/B harness + docs: 1h–2h
- Full-corpus indexing (background): depends on size and rate limits

Notes
- Keep implementation minimal—no backward-compat layers or extra features beyond the retriever swap
- If Gemini must be used end-to-end, prefer building a custom extraction prompt in `graph_rag/index.py` and keep Gemini for generation in `chatbot.py`


