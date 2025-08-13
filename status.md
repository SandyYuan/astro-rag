## Knowledge Graph Implementation Trace

- Implemented `graph_rag/` per plan: `index.py`, `neo4j_client.py`, `inspect.py`.
- Added Neo4j schema:
  - Constraints: `Entity.name`, `Paper.path`, `Claim.id`
  - Full-text indexes (Neo4j 5): `entityFulltext(name, aliases)`, `claimFulltext(text)`, `paperFulltext(title)`
- Indexer (Phase 1) over `.txt` abstracts:
  - LLM extraction (Gemini) for entities, relations, claims; tolerant JSON parsing; stable claim IDs `clm_<hash>`
  - Upserts: `(Entity)-[:MENTIONED_IN]->(Paper)`, `(Claim)-[:SUPPORTED_BY]->(Paper)`, and `(:Claim)-[:ABOUT]->(:Entity)`
  - Entity metadata: `aliases`, `paper_count`, `mention_count`, `top_paper_paths`, `top_claim_ids`
  - Descriptions now derived from top `[:ABOUT]` claims (no generic paper blurbs)
- Inspector prints nodes and edges including `ABOUT` links.
- Retriever (Phase 1):
  - Entity-centric via FT search on entities; aggregates top `[:ABOUT]` claims with sources; no fallback
  - Claim-centric FT path available to group claims by entities when entity search yields nothing
- Local Neo4j running; re-indexed 3 `.txt` files successfully; smoke queries return grounded snippets.

### Recent Changes
- Implemented schema DDL in code (`ensure_schema()`), idempotent
- Extended extraction to include `about_entities`; added `[:ABOUT]` upserts
- Rewrote entity summary logic to use `[:ABOUT]` for `description` and `top_claim_ids`
- Added CLI flag `--update-summaries-only` and explicit `.env` loading in indexer
 - Phase 2 wiring done: `RAG_MODE=neo4j|faiss` toggle added to `chatbot.py`; `GraphRetriever` integrated
 - Switched QA chain call to `invoke` to remove LangChain deprecation warning
 - Fixed FAISS load compatibility (removed deprecated `allow_dangerous_deserialization` and re-saved existing FAISS docstore in current env; no re-embedding). One-time converter removed after use
 - Added clean shutdown for Neo4j driver with `atexit` hook in `graph_rag/neo4j_client.py` and ensured CLI closes driver

### Next Steps
- Improve GraphRetriever intent routing: if query includes parameters/metrics (e.g., `S8`, `σ8`, `Ωm`, “tension”), prefer claim-centric FT search first; else entity-centric
- Strengthen provenance: set `Document.metadata["source"]` to top supporting paper paths (not entity names); include `entity` in metadata
- Ranking/aggregation: rank entities by FT score + claim volume/centrality; aggregate top-k `[:ABOUT]` claims with numeric values and 1–3 sources
- Expand indexing coverage to more `.txt` files so parameter-specific claims (e.g., DES Y3 S8) are captured robustly
- A/B evaluation on ~20 queries (quality, faithfulness, latency) comparing `neo4j` vs `faiss`

### Notes
- No fallbacks: empty results are explicit if no matches/ABOUT claims
- DB hygiene: wipe with `MATCH (n) DETACH DELETE n` for clean runs
