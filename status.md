## Knowledge Graph Implementation Trace

- Implemented `graph_rag/` per plan: `index.py`, `neo4j_client.py`, `inspect.py`.
- Added Neo4j constraints (`Entity.name`, `Paper.path`, `Claim.id`).
- Indexer (Phase 1) over `.txt` abstracts:
  - LLM extraction (Gemini) for entities, relations, claims
  - Tolerant claim parsing; stable IDs `clm_<hash>`
  - Upserts: `(Entity)-[:MENTIONED_IN]->(Paper)`, `(Claim)-[:SUPPORTED_BY]->(Paper)`
  - Entity metadata: `description`, `aliases`, `paper_count`, `mention_count`, `top_paper_paths`, `top_claim_ids`
- Inspector prints nodes and edges with the above metadata.
- Retriever:
  - No fallback. Returns results only when matches found
  - Entity search now uses full-text in code path (pending index creation in DB)
  - Aggregates multiple claim snippets per entity in one document (entity-centric)
- Local Neo4j (Homebrew) running; DB wiped and rebuilt on 3 `.txt` papers; sanity queries work.

### Current Work (in progress)
- Full-text indexes in Neo4j 5 (DB DDL):
  - CREATE FULLTEXT INDEX `entityFulltext` ON `Entity` (name, aliases)
  - CREATE FULLTEXT INDEX `claimFulltext` ON `Claim` (text)
  - CREATE FULLTEXT INDEX `paperFulltext` ON `Paper` (title)
- Extend extraction schema to include `about_entities` per claim; upsert `(:Claim)-[:ABOUT]->(:Entity)`
- Update retriever paths to use:
  - Entity-centric: FT query entities → aggregate top `[:ABOUT]` claims
  - Claim-centric: FT query claims → group by linked entities and return grouped summaries

### Next Steps
- Implement Neo4j 5 DDL in `ensure_schema()` (CREATE FULLTEXT INDEX …) and re-run `--init-schema-only`
- Update indexer to read `about_entities` and upsert `[:ABOUT]` edges; re-index 3 papers
- Verify retrieval for queries: “weak lensing” (entity-centric) and “simulation-based inference” (claim-centric)

### Notes
- No fallbacks implemented (empty results are explicit)
- Graph is persisted in local Neo4j; wipe with `MATCH (n) DETACH DELETE n` for clean runs
