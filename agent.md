## LangGraph QA Agent Integration Plan

Objective: Introduce a LangGraph-based QA agent with conversation memory and tool access to both FAISS RAG and Neo4j GraphRAG. Keep FastAPI (`app.py`) and external API unchanged; make edits minimal and localized.

### 1) Goals
- Multi-turn chat with durable conversation memory
- Retrieval tools: existing FAISS vector store and (optional) Neo4j GraphRAG
- Persona preserved (current system prompt) and sources displayed
- Toggle between legacy flow and agent via env

### 2) Dependencies
- Add/upgrade:
  - `langchain>=0.3,<0.4`
  - `langchain-community>=0.3`
  - `langgraph`
  - Keep `langchain-google-genai` for Gemini LLM/embeddings
  - Ensure `neo4j` (Python driver) if using GraphRAG

Reference (LCEL + chat history, recommended approach):
- LangChain QA with chat history tutorial: https://python.langchain.com/docs/tutorials/qa_chat_history/

### 3) High-level architecture
- Keep
  - FAISS index (`rag_data/index_all`), Gemini LLM/embeddings, FastAPI routes/UI
- Add
  - `agent/graph_app.py`: builds a LangGraph graph with memory + tools
  - Tools: `retrieve_vector` (FAISS), `retrieve_graph` (Neo4j; optional)
  - Checkpointer for session memory (in-memory or SQLite)
- `chatbot.py` gains a small toggle to call the agent instead of legacy chain

### 4) State & memory
- Graph state fields:
  - `messages`: conversation messages (history)
  - `standalone_question`: condensed user query (from follow-up)
  - `context`: merged retrieval context (string)
  - `sources`: list of provenance strings (paper paths, entity names)
- Use LangGraph checkpointer keyed by `session_id` for conversation memory

### 5) Tools
- `retrieve_vector(query)`
  - Wraps current FAISS retriever (`k=5`, `fetch_k=10`, `lambda_mult=0.7`)
  - Returns context text and `sources` (from `Document.metadata['source']`)
- `retrieve_graph(query)` (optional)
  - Calls `graph_rag/neo4j_client.py` `GraphRetriever.get_relevant_documents`
  - Returns synthesized graph context and sources

### 6) Nodes (graph)
- `condense_question` (LCEL):
  - Input: `messages`
  - Output: `standalone_question`
- `route` (heuristic or LLM-based):
  - Decide `vector` / `graph` / `both`
- `retrieve_vector` and `retrieve_graph`:
  - Fetch contexts + sources per route
- `answer` (LCEL):
  - Persona/system prompt + `context` + user input → final answer; emit `sources`

### 7) Implementation steps
1. Create `agent/graph_app.py`:
   - Build FAISS retriever from existing `rag_data/index_all`
   - (Optional) Import `GraphRetriever` from `graph_rag/neo4j_client.py`
   - Define LCEL prompts for `condense_question` and `answer`
   - Implement `retrieve_vector` and `retrieve_graph` tool functions
   - Implement a simple `route` (start with heuristic keywords; upgrade later)
   - Create a LangGraph with nodes above and a checkpointer
   - Expose `run(input_text: str, session_id: str) -> {answer, sources}`
2. Add `CHAT_MODE` toggle in `chatbot.py`:
   - `CHAT_MODE=agent|legacy` (default `legacy`)
   - In agent mode, call `graph_app.run(query, session_id)` instead of legacy QA chain
3. Keep `app.py` unchanged; it continues to call `chatbot.chat()`

### 8) Persona & formatting
- Reuse `AstronomyChatbot.get_system_prompt()` content inside the agent `answer` prompt
- Preserve source formatting (list of distinct `source` strings)
- Keep the UI’s post-processing (removing meta phrases) untouched

### 9) Runbook
1. Install deps:
```
pip install langchain==0.3.* langchain-community==0.3.* langgraph neo4j
```
2. Ensure FAISS index exists:
```
python rag_processor.py
```
3. (Optional) Build Neo4j GraphRAG per `graph.md`
4. Start app in agent mode:
```
CHAT_MODE=agent python app.py
```
5. Switch back (rollback):
```
CHAT_MODE=legacy python app.py
```

### 10) A/B evaluation
- Prepare 15–20 test questions (broad, multi-hop, entity-focused)
- Compare `legacy` vs `agent` on answer quality, citations, latency
- If graph tool improves multi-hop/entity questions, keep it on for those routes

### 11) Deliverables checklist
- [ ] `agent/graph_app.py` (LangGraph graph + tools + memory)
- [ ] `chatbot.py` `CHAT_MODE` toggle and agent call path
- [ ] (Optional) `graph_rag/neo4j_client.py` (if GraphRAG is enabled)
- [ ] Tests/smoke: both modes answer and return sources

### 12) Time estimate
- Agent scaffolding + FAISS tool: 3–5 hours
- Router + GraphRAG tool + tests: 3–6 hours (depending on Neo4j readiness)

### 13) Notes
- ConversationalRetrievalChain is deprecated; use LCEL + LangGraph + `RunnableWithMessageHistory` patterns per the LangChain tutorial above
- Keep changes minimal; no API change to FastAPI; all new logic isolated in `agent/` (and `graph_rag/` if needed)


