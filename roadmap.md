## Roadmap: Post-GraphRAG (Phase 1) Steps

Assumes `graph.md` Phase 1 is completed: Neo4j graph built from existing texts and a `GraphRetriever.get_relevant_documents()` implemented (Cypher-only; no new embeddings).

### Phase 2 — Graph-only chatbot integration
- Goal: Enable a graph-only mode without touching FAISS or the API surface.
- Steps:
  1) Add `RAG_MODE=neo4j|faiss` env toggle in `chatbot.py` constructor.
  2) If `neo4j`, initialize `GraphRetriever` and assign to `self.retriever`; keep current Gemini LLM and QA prompt.
  3) Return `answer` and `sources` as today; no UI changes.
- Deliverables:
  - `chatbot.py` retriever toggle
  - Smoke test: 5–10 multi-hop/entity queries; verify sources
- Success criteria:
  - Correct answers for entity/relationship questions; no regressions in app stability

### Phase 3 — Dual retrieval (graph + FAISS) with fusion
- Goal: Run both retrieval channels per query and fuse contexts before generation.
- Steps:
  1) Keep FAISS retriever as-is; call both FAISS and `GraphRetriever` in parallel.
  2) Fuse results via reciprocal rank fusion (RRF) or simple score normalization; dedupe by `metadata.source`.
  3) Enforce a token budget (e.g., 2–4k tokens) by taking top-N diverse chunks.
  4) Feed merged context into the existing persona prompt.
- Deliverables:
  - A small utility (e.g., `retrieval/fusion.py`) with RRF
  - Toggle to enable fusion mode
- Success criteria:
  - Improved recall on broad/fuzzy queries; no significant latency spikes

### Phase 4 — Vectorize KG summaries (optional, recommended)
- Goal: Add a small vector index for KG artifacts to improve global recall & reduce Cypher fan-out.
- What to index:
  - Entity summaries (name, type, 1–2 sentence description)
  - Claim texts (short declarative statements)
  - Community/topic summaries (if computed)
- Steps:
  1) Add `graph_rag/kg_vector_index.py` to build/update a FAISS (or Chroma) index at `rag_data/kg_index`.
  2) Use the same embedding model as chunks (Gemini `text-embedding-004`) for consistency.
  3) At query-time: retrieve top KG summaries by vectors, then run targeted Cypher expansions for those hits.
  4) Fuse with FAISS chunks and direct graph results as in Phase 3.
- Deliverables:
  - `rag_data/kg_index` and build script
  - Integrated query path using KG vectors → Cypher expansion
- Success criteria:
  - Better coverage for synonym/fuzzy queries; reduced traversal cost

### Phase 5 — Reranking & evaluation
- Goal: Improve final context selection and add measurable evaluation.
- Steps:
  1) Add a cross-encoder reranker (e.g., bge-reranker or MiniLM) after fusion to re-order top contexts.
  2) Create a small evaluation set (15–30 queries) with manual ratings for quality/faithfulness.
  3) Track latency and context length; adjust k/budgets.
- Deliverables:
  - Reranking module and a simple evaluation script/notebook
- Success criteria:
  - Higher rated answers at similar or acceptable latency

### Phase 6 — LangGraph agent (optional, advanced)
- Goal: Multi-turn agent with session memory and iterative tool use.
- Steps:
  1) Build an agent graph (`agent/graph_app.py`) with memory (checkpointer) and tools: FAISS, GraphRetriever, (optional) KG vectors.
  2) Start single-pass (condense → retrieve → answer); then enable ReAct-style tool loops with a max-iterations cap.
  3) Add `CHAT_MODE=agent|legacy` to `chatbot.py`; keep FastAPI unchanged.
- Deliverables:
  - `agent/graph_app.py`, mode toggle, smoke tests
- Success criteria:
  - Better handling of complex, iterative questions across turns

## Notes & Constraints
- No re-embedding of corpus is required for Phase 2–3; Phase 4 embeds only KG summaries (small set).
- Keep FAISS and graph retrieval independent to enable A/B and rollback.
- Centralize any text post-processing (phrase removal) in the backend for consistency.

## Rollback
- Graph-only: set `RAG_MODE=faiss`.
- Dual/fusion: disable the fusion toggle to return to single-channel retrieval.
- Agent: set `CHAT_MODE=legacy`.

## Time Estimates (rough)
- Phase 2: 0.5–1 day
- Phase 3: 0.5–1 day
- Phase 4: 1 day
- Phase 5: 0.5–1 day
- Phase 6: 1–2 days


