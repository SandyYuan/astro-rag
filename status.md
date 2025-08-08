## Knowledge Graph Implementation Trace

- 1) Scaffolding: Added `graph_rag/` with `__init__.py`, `index.py`, `neo4j_client.py` per `graph.md` plan.
- 2) Dependencies: Added `neo4j` Python driver to `requirements.txt`.
- 3) Indexer CLI: Implemented `GraphIndexer` with `--limit`, `--dirs`, and `--init-schema-only` flags.
- 4) Constraints: Added `Entity.name`, `Paper.path`, `Claim.id` constraints creation.
- 5) Retriever: Implemented `GraphRetriever.get_relevant_documents()` with simple entity search and neighborhood claims.
- 5.1) Removed global fallback path to comply with "no fallbacks" rule. Now returns empty when no entity match.
- 6) Sanity checks: Pending — run `python -m graph_rag.index --init-schema-only` after setting `NEO4J_*`.
 - 6) Sanity checks: CLI help verified for indexer and retriever inside `mcp` env. Next: set `NEO4J_*` in `.env` and run `python -m graph_rag.index --init-schema-only`.

### Blocker
- **Neo4j not reachable**: `bolt://localhost:7687` connection refused. Docker daemon is not running on this machine, so a local Neo4j container cannot be started yet. No Aura credentials provided.

### Unblock options (pick one)
- Start Docker Desktop, then run: `docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5`
- Provide Aura `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Or install local Neo4j via Homebrew and start the service


