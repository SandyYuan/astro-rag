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

### Next Steps
- Phase 2 wiring: add `RAG_MODE=neo4j|faiss` toggle in `chatbot.py` and smoke test end-to-end
- Index a larger subset (or full corpus) to improve coverage (e.g., weak lensing)
- A/B evaluation on ~20 queries (answer quality, sources, latency)

### Notes
- No fallbacks: empty results are explicit if no matches/ABOUT claims
- DB hygiene: wipe with `MATCH (n) DETACH DELETE n` for clean runs
