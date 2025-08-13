import os
import logging
import atexit
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.schema import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class GraphRetriever:
    """Neo4j-backed retriever returning LangChain Documents.

    Phase 1: very simple entity phrase search and k-hop neighborhood lookup.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.uri = _get_env("NEO4J_URI")
        self.user = _get_env("NEO4J_USER")
        self.password = _get_env("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # Ensure clean shutdown to avoid driver __del__ warnings at interpreter exit
        atexit.register(self._safe_close)

    def _safe_close(self) -> None:
        try:
            if getattr(self, "driver", None) is not None:
                self.driver.close()
        except BaseException:
            # Ignore shutdown-time errors
            pass

    def close(self) -> None:
        self._safe_close()

    def _search_entities(self, phrase: str) -> List[Dict[str, Any]]:
        # Full-text search over entity index
        query = (
            "CALL db.index.fulltext.queryNodes('entityFulltext', $q) YIELD node, score "
            "RETURN node.name AS name, node.type AS type, score ORDER BY score DESC LIMIT 10"
        )
        with self.driver.session() as session:
            records = session.run(query, {"q": phrase}).data()
        return records

    def _fetch_entity_claims(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        # Prefer explicit ABOUT edges, get arXiv URLs when available
        query = (
            "MATCH (e:Entity {name: $name})<-[:ABOUT]-(c:Claim) "
            "OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(p:Paper) "
            "RETURN c.text AS claim, "
            "collect(DISTINCT coalesce(p.arxiv_url, p.path))[..3] AS sources LIMIT $limit"
        )
        with self.driver.session() as session:
            records = session.run(query, {"name": name, "limit": limit}).data()
        return records

    def _search_claims(self, phrase: str, limit: int = 20) -> List[Dict[str, Any]]:
        query = (
            "CALL db.index.fulltext.queryNodes('claimFulltext', $q) YIELD node, score "
            "OPTIONAL MATCH (node)-[:ABOUT]->(e:Entity) "
            "RETURN node.text AS claim, collect(DISTINCT e.name)[..3] AS entities, score "
            "ORDER BY score DESC LIMIT $limit"
        )
        with self.driver.session() as session:
            return session.run(query, {"q": phrase, "limit": limit}).data()

    def get_relevant_documents(self, query_str: str) -> List[Document]:
        docs: List[Document] = []
        # Try entity-centric aggregation
        entities = self._search_entities(query_str)
        if entities:
            # Aggregate per-entity: 1 document per entity, include top claims
            for ent in entities[: self.k]:
                name = ent["name"]
                claims = self._fetch_entity_claims(name, limit=5)
                lines = [f"Entity: {name}"]
                # Collect all sources from claims for provenance
                all_sources = []
                for c in claims:
                    if c.get("claim"):
                        lines.append(f"- {c['claim']} (src: {', '.join(c.get('sources', []) or [])})")
                        all_sources.extend(c.get('sources', []) or [])
                
                page_content = "\n".join(lines)
                # Use first paper source as metadata["source"], entity name in metadata["entity"]
                source = all_sources[0] if all_sources else name
                metadata = {"source": source, "entity": name}
                docs.append(Document(page_content=page_content, metadata=metadata))
            return docs

        # Claim-centric: group by entities (keep entity name as source since no direct paper access)
        claim_hits = self._search_claims(query_str, limit=20)
        if not claim_hits:
            return []
        # Build one document summarizing top claims grouped by entities
        grouped: Dict[str, List[str]] = {}
        for h in claim_hits[: 3 * self.k]:
            claim = h.get("claim")
            ent_list = h.get("entities") or ["(unknown entity)"]
            for en in ent_list:
                grouped.setdefault(en, []).append(claim)
        # Take top-k entities and 2 claims each
        for en, cl in list(grouped.items())[: self.k]:
            lines = [f"Entity: {en}"] + [f"- {c}" for c in cl[:2] if c]
            docs.append(Document(page_content="\n".join(lines), metadata={"source": en}))
        return docs


def _parse_args(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test GraphRetriever")
    parser.add_argument("--q", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    retriever = GraphRetriever(k=args.k)
    try:
        docs = retriever.get_relevant_documents(args.q)
        for i, d in enumerate(docs, 1):
            print(f"[#{i}] {d.metadata.get('source')}:\n{d.page_content}\n")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()


