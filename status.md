## Knowledge Graph Implementation Trace

- 1) Scaffolding: Added `graph_rag/` with `__init__.py`, `index.py`, `neo4j_client.py` per `graph.md` plan.
- 2) Dependencies: Added `neo4j` Python driver to `requirements.txt`.
- 3) Indexer CLI: Implemented `GraphIndexer` with `--limit`, `--dirs`, and `--init-schema-only` flags.
- 4) Constraints: Added `Entity.name`, `Paper.path`, `Claim.id` constraints creation.
- 5) Retriever: Implemented `GraphRetriever.get_relevant_documents()` with simple entity search and neighborhood claims.
- 6) Sanity checks: Pending — run `python -m graph_rag.index --init-schema-only` after setting `NEO4J_*`.
 - 6) Sanity checks: CLI help verified for indexer and retriever inside `mcp` env. Next: set `NEO4J_*` in `.env` and run `python -m graph_rag.index --init-schema-only`.


