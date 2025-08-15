import os
import json
import glob
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional
import hashlib
import re

from dotenv import load_dotenv
from pathlib import Path
from neo4j import GraphDatabase

from llm_provider import LLMProvider

# Load .env explicitly from repository root to ensure availability in all run contexts
load_dotenv(dotenv_path=str(Path(__file__).resolve().parents[1] / ".env"), override=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str


class GraphIndexer:
    """Builds a minimal knowledge graph in Neo4j from text files.

    Phase 1: deterministic upserts of Paper, Entity, Claim based on LLM extraction.
    """

    def __init__(
        self,
        neo4j_config: Optional[Neo4jConfig] = None,
        input_dirs: Optional[List[str]] = None,
        llm_provider: Optional[LLMProvider] = None,
        limit_files: Optional[int] = None,
        skip_existing: bool = True,
    ) -> None:
        self.neo4j = neo4j_config or Neo4jConfig(
            uri=_get_env("NEO4J_URI"),
            user=_get_env("NEO4J_USER"),
            password=_get_env("NEO4J_PASSWORD"),
        )
        self.input_dirs = input_dirs or ["papers", "papers_np"]
        self.limit_files = limit_files
        self.skip_existing = skip_existing
        # Reuse existing provider for extraction. Keep deterministic settings.
        # Do NOT initialize LLM until extraction time, so schema-only flows work without GOOGLE_API_KEY.
        self.llm_provider = llm_provider
        self.llm = None

        self.driver = GraphDatabase.driver(self.neo4j.uri, auth=(self.neo4j.user, self.neo4j.password))

    # ---------------------------
    # Schema and constraints
    # ---------------------------
    def ensure_schema(self) -> None:
        """Create constraints and Neo4j 5 full-text indexes (idempotent)."""
        constraint_ddl = [
            """
            CREATE CONSTRAINT entity_name IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT paper_path IF NOT EXISTS
            FOR (p:Paper) REQUIRE p.path IS UNIQUE
            """,
            """
            CREATE CONSTRAINT claim_id IF NOT EXISTS
            FOR (c:Claim) REQUIRE c.id IS UNIQUE
            """,
        ]

        fulltext_ddl = [
            # Index both single string and array property via ON EACH
            """
            CREATE FULLTEXT INDEX entityFulltext IF NOT EXISTS
            FOR (e:Entity) ON EACH [e.name, e.aliases]
            """,
            """
            CREATE FULLTEXT INDEX claimFulltext IF NOT EXISTS
            FOR (c:Claim) ON EACH [c.text]
            """,
            """
            CREATE FULLTEXT INDEX paperFulltext IF NOT EXISTS
            FOR (p:Paper) ON EACH [p.title]
            """,
        ]

        with self.driver.session() as session:
            for stmt in constraint_ddl:
                session.run(stmt)
            for stmt in fulltext_ddl:
                session.run(stmt)
        logger.info("Neo4j constraints and full-text indexes ensured")

    # ---------------------------
    # File loading
    # ---------------------------
    def list_input_texts(self) -> List[str]:
        files: List[str] = []
        for d in self.input_dirs:
            if os.path.isdir(d):
                files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))
        if self.limit_files:
            files = files[: self.limit_files]
        logger.info(f"Found {len(files)} .txt files for indexing")
        return files

    def _paper_already_processed(self, path: str) -> bool:
        """Return True if the paper already has at least one supported claim in the graph.

        This is used to skip re-processing papers on resume to avoid extra LLM calls.
        """
        query = (
            "MATCH (p:Paper {path: $path})<-[:SUPPORTED_BY]-(:Claim) "
            "RETURN count(*) AS cnt"
        )
        with self.driver.session() as session:
            rec = session.run(query, {"path": path}).single()
            cnt = rec[0] if rec is not None else 0
        return (cnt or 0) > 0

    # ---------------------------
    # LLM extraction
    # ---------------------------
    def extract_graph_items(self, text: str) -> Dict[str, Any]:
        """Use the existing LLM to extract simple entities/relations/claims.

        Output schema:
        {
          "entities": [{"name": str, "type": str|null}],
          "relations": [{"source": str, "target": str, "relation_type": str, "evidence": str|null}],
          "claims": [{"id": str, "text": str, "confidence": float|null}]
        }
        """
        # Lazy LLM init to avoid requiring GOOGLE_API_KEY for schema-only operations
        if self.llm is None:
            provider = self.llm_provider or LLMProvider()
            model_override = os.getenv("GRAPH_RAG_TEXT_MODEL", "gemini-2.5-flash")
            if model_override:
                self.llm = provider.get_llm(temperature=0.1, model_name=model_override)
            else:
                self.llm = provider.get_llm(temperature=0.1)

        prompt = (
            "Extract key entities (names and optional types), simple typed relations between them, "
            "and 1-5 short, declarative claims from the text below. Return strict JSON with keys: "
            "entities, relations, claims. Each claim must be an object with fields {text, id (optional), confidence (optional), about_entities (array of canonical entity names)}. "
            "If unsure, include best-effort claims and set low confidence; do not return an empty 'claims' list.\n\n"
            "TEXT:\n" + text[:8000]
        )
        raw = self.llm(prompt)
        try:
            data = json.loads(raw)
        except Exception:
            # If model responded with extra prose, try to find JSON block
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])
            else:
                raise ValueError("LLM did not return valid JSON for extraction")

        # Normalize
        entities: List[Dict[str, Any]] = []
        for e in data.get("entities", []):
            if isinstance(e, str):
                name = e.strip()
                if name:
                    entities.append({"name": name, "type": None})
            elif isinstance(e, dict):
                name = (e.get("name") or "").strip()
                if name:
                    etype = (e.get("type") or "").strip() or None
                    entities.append({"name": name, "type": etype})
        relations = [
            {
                "source": r.get("source", "").strip(),
                "target": r.get("target", "").strip(),
                "relation_type": r.get("relation_type", "").strip(),
                "evidence": (r.get("evidence") or "").strip() or None,
            }
            for r in data.get("relations", [])
            if r.get("source") and r.get("target") and r.get("relation_type")
        ]
        def _stable_claim_id(text: str) -> str:
            h = hashlib.blake2s(text.encode("utf-8"), digest_size=8).hexdigest()  # 16 hex chars
            return f"clm_{h}"

        claims: List[Dict[str, Any]] = []
        for c in data.get("claims", []):
            if isinstance(c, str):
                text = c.strip()
                if not text:
                    continue
                cid = _stable_claim_id(text)
                claims.append({"id": cid, "text": text, "confidence": None, "about_entities": []})
            elif isinstance(c, dict):
                raw_text = c.get("text")
                # Coerce non-strings to strings safely
                text = (str(raw_text) if raw_text is not None else "").strip()
                if not text:
                    continue
                cid_in = (c.get("id") or "").strip()
                # Always normalize to our stable format for consistency
                cid = _stable_claim_id(text)
                conf = c.get("confidence")
                about_list = c.get("about_entities") or c.get("about") or []
                if isinstance(about_list, str):
                    about_list = [about_list]
                about_entities = []
                for ae in about_list:
                    try:
                        s = str(ae).strip()
                        if s:
                            about_entities.append(s)
                    except Exception:
                        continue
                claims.append({"id": cid, "text": text, "confidence": conf, "about_entities": about_entities})
        return {"entities": entities, "relations": relations, "claims": claims}

    # ---------------------------
    # Upsert into Neo4j
    # ---------------------------
    @staticmethod
    def _normalize_name(name: str) -> str:
        base = name.lower().strip()
        base = re.sub(r"\s+", " ", base)
        base = re.sub(r"[^a-z0-9 _:\-/]", "", base)
        return base

    @staticmethod
    def _generate_aliases(name: str) -> List[str]:
        aliases = set()
        base = name.strip()
        aliases.add(base)
        aliases.add(base.lower())
        aliases.add(re.sub(r"[\-_:]", " ", base))
        aliases = {a for a in aliases if a}
        return list(aliases)[:5]

    @staticmethod
    def _extract_arxiv_url(text: str) -> Optional[str]:
        """Extract arXiv URL from paper text."""
        for line in text.splitlines()[:20]:  # Check first 20 lines
            if line.startswith("arXiv URL:"):
                url = line.split(":", 1)[1].strip()
                return url if url.startswith("http") else None
        return None

    @staticmethod
    def _normalize_relation_type(relation_type: str) -> str:
        """Convert relation_type to valid Neo4j relationship name.
        
        Common types get their own relationship types, others fall back to RELATES_TO.
        """
        # Common scientific relation types
        common_types = {
            "measures": "MEASURES",
            "predicts": "PREDICTS", 
            "constrains": "CONSTRAINS",
            "conflicts_with": "CONFLICTS_WITH",
            "supports": "SUPPORTS",
            "uses": "USES",
            "analyzes": "ANALYZES",
            "compares": "COMPARES",
            "extends": "EXTENDS",
            "validates": "VALIDATES",
            "improves": "IMPROVES",
            "based_on": "BASED_ON",
            "part_of": "PART_OF",
            "contains": "CONTAINS",
            "derives_from": "DERIVES_FROM"
        }
        
        normalized = relation_type.lower().strip().replace(" ", "_").replace("-", "_")
        return common_types.get(normalized, "RELATES_TO")

    def upsert_document(self, path: str, title: Optional[str], items: Dict[str, Any], text: str = "") -> None:
        arxiv_url = self._extract_arxiv_url(text) if text else None
        with self.driver.session() as session:
            # Upsert Paper with arXiv URL
            session.run(
                """
                MERGE (p:Paper {path: $path})
                ON CREATE SET p.title = $title, p.arxiv_url = $arxiv_url
                ON MATCH SET p.title = coalesce($title, p.title), p.arxiv_url = coalesce($arxiv_url, p.arxiv_url)
                """,
                {"path": path, "title": title, "arxiv_url": arxiv_url},
            )

            # Entities
            # Deduplicate entity names per document
            seen_entities: Dict[str, Optional[str]] = {}
            for e in items.get("entities", []):
                ename = e.get("name") if isinstance(e, dict) else str(e)
                if not ename:
                    continue
                if ename not in seen_entities:
                    seen_entities[ename] = (e.get("type") if isinstance(e, dict) else None)
            for ename, etype in seen_entities.items():
                norm_name = self._normalize_name(ename)
                aliases = self._generate_aliases(ename)
                session.run(
                    """
                    MERGE (en:Entity {name: $name})
                    ON CREATE SET en.type = $type, en.normalized_name = $norm, en.aliases = $aliases
                    ON MATCH SET en.type = coalesce($type, en.type),
                                  en.normalized_name = coalesce(en.normalized_name, $norm),
                                  en.aliases = coalesce(en.aliases, $aliases)
                    WITH en
                    MATCH (p:Paper {path: $paper})
                    MERGE (en)-[m:MENTIONED_IN]->(p)
                    """,
                    {"name": ename, "type": etype, "paper": path, "norm": norm_name, "aliases": aliases},
                )

            # Claims
            for c in items.get("claims", []):
                about_entities = c.get("about_entities") or []
                session.run(
                    """
                    MERGE (cl:Claim {id: $id})
                    ON CREATE SET cl.text = $text, cl.confidence = $conf
                    ON MATCH SET cl.text = coalesce($text, cl.text), cl.confidence = coalesce($conf, cl.confidence)
                    WITH cl
                    MATCH (p:Paper {path: $paper})
                    MERGE (cl)-[:SUPPORTED_BY]->(p)
                    WITH cl
                    UNWIND $about AS ename
                    MERGE (en:Entity {name: ename})
                    MERGE (cl)-[:ABOUT]->(en)
                    """,
                    {
                        "id": c["id"],
                        "text": c["text"],
                        "conf": c.get("confidence"),
                        "paper": path,
                        "about": about_entities,
                    },
                )

            # Relations - use typed relationships
            for r in items.get("relations", []):
                rel_type = self._normalize_relation_type(r["relation_type"])
                
                # Use dynamic relationship type in Cypher
                if rel_type == "RELATES_TO":
                    # For generic relations, store original type as property
                    session.run(
                        """
                        MERGE (s:Entity {name: $source})
                        MERGE (t:Entity {name: $target})
                        MERGE (s)-[rel:RELATES_TO {relation_type: $rtype}]->(t)
                        ON CREATE SET rel.evidence = $evidence
                        ON MATCH SET rel.evidence = coalesce($evidence, rel.evidence)
                        """,
                        {
                            "source": r["source"],
                            "target": r["target"],
                            "rtype": r["relation_type"],
                            "evidence": r.get("evidence"),
                        },
                    )
                else:
                    # For common types, create typed relationship
                    query = f"""
                        MERGE (s:Entity {{name: $source}})
                        MERGE (t:Entity {{name: $target}})
                        MERGE (s)-[rel:{rel_type}]->(t)
                        ON CREATE SET rel.evidence = $evidence, rel.original_type = $rtype
                        ON MATCH SET rel.evidence = coalesce($evidence, rel.evidence)
                        """
                    session.run(query, {
                        "source": r["source"],
                        "target": r["target"],
                        "rtype": r["relation_type"],
                        "evidence": r.get("evidence"),
                    })

    # ---------------------------
    # Public API
    # ---------------------------
    def run(self) -> None:
        self.ensure_schema()
        files = self.list_input_texts()
        for path in files:
            try:
                if self.skip_existing and self._paper_already_processed(path):
                    logger.info(f"Skipping (already processed): {path}")
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                # Prefer actual title from .txt if present (line starting with 'Title:')
                title = None
                for line in text.splitlines():
                    if line.strip():
                        if line.lower().startswith("title:"):
                            title = line.split(":", 1)[1].strip()
                        break
                if not title:
                    title = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
                items = self.extract_graph_items(text)
                self.upsert_document(path=path, title=title, items=items, text=text)
                logger.info(f"Indexed: {path}")
            except Exception as e:
                logger.error(f"Failed indexing {path}: {e}")

        # Post-process entity summaries (counts, top references, description)
        self.update_entity_summaries()

    def update_entity_summaries(self) -> None:
        with self.driver.session() as session:
            # Clear stale descriptions before recomputing from ABOUT edges
            session.run(
                """
                MATCH (e:Entity)
                SET e.description = NULL
                """
            )
            # mention_count and paper_count
            session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:MENTIONED_IN]->(p:Paper)
                WITH e, count(DISTINCT p) AS pc
                SET e.paper_count = pc,
                    e.mention_count = pc
                """
            )
            # top_paper_paths by number of claims supported in that paper (desc)
            session.run(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(p:Paper)
                OPTIONAL MATCH (c:Claim)-[:SUPPORTED_BY]->(p)
                WITH e, p, count(c) AS cc
                ORDER BY e.name, cc DESC
                WITH e, collect({path: p.path, cc: cc}) AS plist
                SET e.top_paper_paths = [x IN plist[..3] | x.path]
                """
            )
            # top_claim_ids for entity
            session.run(
                """
                MATCH (e:Entity)<-[:ABOUT]-(c:Claim)
                WITH e, c, count(*) AS freq
                ORDER BY e.name, freq DESC
                WITH e, collect({id: c.id, f: freq}) AS cl
                SET e.top_claim_ids = [x IN cl[..5] | x.id]
                """
            )
            # description from top claim text (first only to avoid string concat issues)
            session.run(
                """
                MATCH (e:Entity)<-[:ABOUT]-(c:Claim)
                WITH e, c, count(*) AS freq
                ORDER BY e.name, freq DESC
                WITH e, collect(c.text) AS texts
                SET e.description = CASE WHEN size(texts) > 0 THEN texts[0] ELSE e.description END
                """
            )


def _parse_args(argv: Optional[List[str]] = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Graph indexer for Neo4j from text corpus")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of .txt files to index")
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["papers", "papers_np"],
        help="Input directories to scan for .txt files",
    )
    parser.add_argument(
        "--init-schema-only",
        action="store_true",
        help="Only create constraints/indexes and exit",
    )
    parser.add_argument(
        "--update-summaries-only",
        action="store_true",
        help="Only recompute entity summaries/descriptions and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    indexer = GraphIndexer(input_dirs=args.dirs, limit_files=args.limit)
    if args.init_schema_only:
        indexer.ensure_schema()
        logger.info("Schema initialized. Exiting as requested.")
        return
    if args.update_summaries_only:
        indexer.update_entity_summaries()
        logger.info("Entity summaries updated. Exiting as requested.")
        return
    indexer.run()


if __name__ == "__main__":
    main()


