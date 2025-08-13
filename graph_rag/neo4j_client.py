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

    def _is_quality_entity(self, entity_name: str) -> bool:
        """Filter out low-quality entities: generic references, paper titles, and poor extractions."""
        name = entity_name.strip()
        
        # Filter out generic references
        generic_patterns = [
            'the study', 'this paper', 'the paper', 'this work', 'the work',
            'the analysis', 'the research', 'the method', 'the approach',
            'the model', 'the survey', 'the data', 'the results',
            'the collaboration', 'the team', 'the authors'
        ]
        
        if name.lower() in generic_patterns:
            return False
            
        # Filter out paper titles (contain "Results:" or are very long)
        if 'Results:' in name or 'results:' in name:
            return False
            
        if len(name) > 80:  # Likely paper titles
            return False
            
        # Filter out overly procedural entities
        procedural_patterns = [
            'constraints on', 'analysis of', 'study of', 'measurement of',
            'observations of', 'detection of', 'survey of'
        ]
        
        name_lower = name.lower()
        if any(name_lower.startswith(pattern) for pattern in procedural_patterns):
            if len(name) > 50:  # Only filter if also long
                return False
                
        # Filter out entities that are mostly punctuation or formatting
        # But allow scientific parameters like S8, H0, etc.
        cleaned_name = name.replace(' ', '').replace('-', '').replace('_', '')
        if len(cleaned_name) < 2:  # Changed from 3 to 2
            return False
            
        # Allow scientific parameters (contain numbers and letters)
        if any(char.isdigit() for char in cleaned_name) and any(char.isalpha() for char in cleaned_name):
            if len(cleaned_name) >= 2:  # S8, H0, etc. are valid
                return True
            
        # For non-parameter entities, require at least 3 meaningful characters
        if len(cleaned_name) < 3 and not any(char.isdigit() for char in cleaned_name):
            return False
            
        return True

    def _deduplicate_claims(self, claims_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove exact and semantic duplicates from claims."""
        if not claims_data:
            return claims_data
            
        deduplicated = []
        seen_exact = set()
        seen_semantic = set()
        
        for claim_data in claims_data:
            claim_text = claim_data.get('claim', '').strip()
            if not claim_text:
                continue
                
            # Check exact duplicates (case-insensitive, normalized)
            normalized_text = claim_text.lower().strip()
            if normalized_text in seen_exact:
                continue
                
            # Check semantic duplicates (same key information)
            semantic_key = self._get_semantic_key(claim_text)
            if semantic_key and semantic_key in seen_semantic:
                continue
                
            # Add to seen sets and include in results
            seen_exact.add(normalized_text)
            if semantic_key:
                seen_semantic.add(semantic_key)
            deduplicated.append(claim_data)
            
        return deduplicated
        
    def _get_semantic_key(self, claim_text: str) -> str:
        """Generate a semantic key for duplicate detection."""
        text_lower = claim_text.lower()
        
        # DES-Planck conflict claims
        if 'des' in text_lower and 'planck' in text_lower:
            if any(word in text_lower for word in ['conflict', 'disagree', 'tension', 'lower', 'favors']):
                return 'des_planck_conflict'
                
        # S8 measurement claims - extract the actual measurement if present
        if 's_8' in text_lower or 's8' in text_lower:
            # Look for specific measurements like "S_8 = 0.792±0.012"
            import re
            s8_pattern = r's_?8\s*=\s*([0-9.±+\-\s]+)'
            match = re.search(s8_pattern, text_lower)
            if match:
                measurement = match.group(1).strip()
                return f's8_measurement_{measurement}'
                
            # General S8 measurement without specific value
            if any(word in text_lower for word in ['yielded', 'resulted', 'found', 'measured', 'constrained']):
                return 's8_measurement_general'
                
        # Consistency/agreement claims
        if any(word in text_lower for word in ['consistent', 'agreement', 'agrees']):
            if 'planck' in text_lower or 'cmb' in text_lower:
                return 'consistency_planck'
                
        return None  # No semantic grouping identified

    def _fetch_entity_claims(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        # Enhanced method: get claims from entity AND its 1-hop semantic neighbors
        records = []
        
        with self.driver.session() as session:
            # 1. Get direct claims about the entity
            direct_query = (
                "MATCH (e:Entity {name: $name})<-[:ABOUT]-(c:Claim) "
                "OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(p:Paper) "
                "RETURN c.text AS claim, "
                "collect(DISTINCT coalesce(p.arxiv_url, p.path))[..3] AS sources, "
                "'direct' AS claim_type, $name AS entity_context "
                "LIMIT $direct_limit"
            )
            direct_records = session.run(direct_query, {
                "name": name, 
                "direct_limit": limit // 2  # Reserve half for neighbors
            }).data()
            records.extend(direct_records)
            
            # 2. Get claims from semantic neighbors
            neighbor_query = (
                "MATCH (e:Entity {name: $name})-[r]-(neighbor:Entity) "
                "WHERE type(r) IN ['MEASURES', 'PREDICTS', 'USES', 'CONSTRAINS', 'SUPPORTS', 'ANALYZES'] "
                "MATCH (neighbor)<-[:ABOUT]-(c:Claim) "
                "OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(p:Paper) "
                "RETURN c.text AS claim, "
                "collect(DISTINCT coalesce(p.arxiv_url, p.path))[..3] AS sources, "
                "'neighbor' AS claim_type, neighbor.name AS entity_context "
                "LIMIT $neighbor_limit"
            )
            neighbor_records = session.run(neighbor_query, {
                "name": name,
                "neighbor_limit": limit // 2  # Use remaining half for neighbors
            }).data()
            records.extend(neighbor_records)
            
        return records

    def _fetch_paper_context(self, paper_paths: List[str], limit_per_paper: int = 3) -> List[Dict[str, Any]]:
        """Get additional relevant claims from the same papers, filtered for quality entities."""
        if not paper_paths:
            return []
            
        records = []
        with self.driver.session() as session:
            for paper_path in paper_paths[:5]:  # Limit to 5 papers max
                query = (
                    "MATCH (p:Paper {path: $paper_path})<-[:SUPPORTED_BY]-(c:Claim)-[:ABOUT]->(e:Entity) "
                    "RETURN c.text AS claim, "
                    "coalesce(p.arxiv_url, p.path) AS source, "
                    "'paper_context' AS claim_type, $paper_path AS entity_context, "
                    "e.name AS about_entity "
                    "LIMIT $limit"
                )
                paper_records = session.run(query, {
                    "paper_path": paper_path,
                    "limit": limit_per_paper * 2  # Get more to filter
                }).data()
                
                # Filter for quality entities only
                filtered_records = [
                    record for record in paper_records 
                    if self._is_quality_entity(record.get('about_entity', ''))
                ][:limit_per_paper]  # Take only the limit after filtering
                
                records.extend(filtered_records)
                
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
        # Filter out low-quality entities
        quality_entities = [ent for ent in entities if self._is_quality_entity(ent["name"])]
        
        if quality_entities:
            # Aggregate per-entity: 1 document per entity, include claims from entity + neighbors + paper context
            for ent in quality_entities[: self.k]:
                name = ent["name"]
                claims = self._fetch_entity_claims(name, limit=15)  # Get more to account for deduplication
                
                # Apply deduplication to all claims
                claims = self._deduplicate_claims(claims)
                
                lines = [f"Entity: {name}"]
                
                # Collect all sources from claims for provenance and paper context
                all_sources = []
                paper_paths = set()  # Track unique paper paths
                direct_claims = []
                neighbor_claims = []
                
                # Separate direct vs neighbor claims and collect paper paths
                for c in claims:
                    if c.get("claim"):
                        claim_type = c.get("claim_type", "direct")
                        entity_context = c.get("entity_context", name)
                        sources = c.get('sources', []) or []
                        
                        if claim_type == "direct":
                            direct_claims.append(f"- {c['claim']} (src: {', '.join(sources)})")
                        else:
                            # Include context for neighbor claims
                            neighbor_claims.append(f"- [{entity_context}] {c['claim']} (src: {', '.join(sources)})")
                        
                        all_sources.extend(sources)
                        # Collect paper paths for context (filter out arXiv URLs)
                        for source in sources:
                            if source and not source.startswith('http'):
                                paper_paths.add(source)
                
                # Add direct claims first, then neighbor claims
                lines.extend(direct_claims)
                if neighbor_claims:
                    lines.append("Related entities:")
                    lines.extend(neighbor_claims)
                
                # Add paper context from the same papers
                if paper_paths:
                    paper_context = self._fetch_paper_context(list(paper_paths), limit_per_paper=2)
                    # Deduplicate paper context as well
                    paper_context = self._deduplicate_claims(paper_context)
                    if paper_context:
                        lines.append("Additional context from same papers:")
                        for pc in paper_context[:6]:  # Limit additional context
                            if pc.get("claim"):
                                paper_name = pc.get('entity_context', '').split('/')[-1] if pc.get('entity_context') else 'unknown'
                                lines.append(f"- [{paper_name}] {pc['claim']}")
                
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


