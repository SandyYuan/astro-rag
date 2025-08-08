"""GraphRAG minimal package.

Provides:
- Indexer to build a Neo4j knowledge graph from existing text files
- GraphRetriever to fetch context for RAG

Environment variables required for Neo4j:
- NEO4J_URI
- NEO4J_USER
- NEO4J_PASSWORD
"""

__all__ = [
    "GraphIndexer",
    "GraphRetriever",
]

from .index import GraphIndexer  # noqa: E402
from .neo4j_client import GraphRetriever  # noqa: E402


