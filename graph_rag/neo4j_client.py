import os
import logging
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

    def _search_entities(self, phrase: str) -> List[Dict[str, Any]]:
        # Simple exact/contains search; fulltext index can be added later if desired
        query = (
            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q) "
            "RETURN e.name AS name, e.type AS type LIMIT 10"
        )
        with self.driver.session() as session:
            records = session.run(query, {"q": phrase}).data()
        return records

    def _fetch_neighborhood_claims(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        # Use mentions to collect claims supported by papers that mention the entity
        query = (
            "MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(p:Paper) "
            "OPTIONAL MATCH (c:Claim)-[:SUPPORTED_BY]->(p) "
            "WITH c, p WHERE c IS NOT NULL "
            "RETURN c.text AS claim, collect(DISTINCT p.path)[..3] AS sources LIMIT $limit"
        )
        with self.driver.session() as session:
            records = session.run(query, {"name": name, "limit": limit}).data()
        return records

    def get_relevant_documents(self, query_str: str) -> List[Document]:
        # Heuristic: if the query contains a capitalized token likely to be an entity, prefer local search
        candidate_entities = self._search_entities(query_str)
        docs: List[Document] = []
        if candidate_entities:
            # Use the best-matching entity
            entity_name = candidate_entities[0]["name"]
            claims = self._fetch_neighborhood_claims(entity_name, limit=30)
            for c in claims[: self.k]:
                claim_text = c.get('claim')
                if not claim_text:
                    continue
                sources = c.get('sources', []) or []
                page_content = (
                    f"Entity: {entity_name}\n"
                    f"Claim: {claim_text}\n"
                    f"Sources: {', '.join(sources)}"
                )
                docs.append(Document(page_content=page_content, metadata={"source": entity_name}))
        # No fallback path; if no entity match, return empty list
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
    docs = retriever.get_relevant_documents(args.q)
    for i, d in enumerate(docs, 1):
        print(f"[#{i}] {d.metadata.get('source')}:\n{d.page_content}\n")


if __name__ == "__main__":
    main()


