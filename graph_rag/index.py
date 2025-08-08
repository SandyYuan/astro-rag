import os
import json
import glob
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional
import hashlib

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_provider import LLMProvider

load_dotenv()

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
    ) -> None:
        self.neo4j = neo4j_config or Neo4jConfig(
            uri=_get_env("NEO4J_URI"),
            user=_get_env("NEO4J_USER"),
            password=_get_env("NEO4J_PASSWORD"),
        )
        self.input_dirs = input_dirs or ["papers", "papers_np"]
        self.limit_files = limit_files
        # Reuse existing provider for extraction. Keep deterministic settings.
        # Do NOT initialize LLM until extraction time, so schema-only flows work without GOOGLE_API_KEY.
        self.llm_provider = llm_provider
        self.llm = None

        self.driver = GraphDatabase.driver(self.neo4j.uri, auth=(self.neo4j.user, self.neo4j.password))

    # ---------------------------
    # Schema and constraints
    # ---------------------------
    def ensure_schema(self) -> None:
        """Create minimal constraints. Safe to run multiple times."""
        cypher_statements = [
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
        with self.driver.session() as session:
            for stmt in cypher_statements:
                session.run(stmt)
        logger.info("Neo4j constraints ensured")

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
            "entities, relations, claims. Each claim must be an object with fields {text, id (optional), confidence (optional)}. "
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
                claims.append({"id": cid, "text": text, "confidence": None})
            elif isinstance(c, dict):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                cid_in = (c.get("id") or "").strip()
                # Always normalize to our stable format for consistency
                cid = _stable_claim_id(text)
                conf = c.get("confidence")
                claims.append({"id": cid, "text": text, "confidence": conf})
        return {"entities": entities, "relations": relations, "claims": claims}

    # ---------------------------
    # Upsert into Neo4j
    # ---------------------------
    def upsert_document(self, path: str, title: Optional[str], items: Dict[str, Any]) -> None:
        with self.driver.session() as session:
            # Upsert Paper
            session.run(
                """
                MERGE (p:Paper {path: $path})
                ON CREATE SET p.title = $title
                ON MATCH SET p.title = coalesce($title, p.title)
                """,
                {"path": path, "title": title},
            )

            # Entities
            for e in items.get("entities", []):
                session.run(
                    """
                    MERGE (en:Entity {name: $name})
                    ON CREATE SET en.type = $type
                    ON MATCH SET en.type = coalesce($type, en.type)
                    WITH en
                    MATCH (p:Paper {path: $paper})
                    MERGE (en)-[m:MENTIONED_IN]->(p)
                    """,
                    {"name": e["name"], "type": e.get("type"), "paper": path},
                )

            # Claims
            for c in items.get("claims", []):
                session.run(
                    """
                    MERGE (cl:Claim {id: $id})
                    ON CREATE SET cl.text = $text, cl.confidence = $conf
                    ON MATCH SET cl.text = coalesce($text, cl.text), cl.confidence = coalesce($conf, cl.confidence)
                    WITH cl
                    MATCH (p:Paper {path: $paper})
                    MERGE (cl)-[:SUPPORTED_BY]->(p)
                    """,
                    {"id": c["id"], "text": c["text"], "conf": c.get("confidence"), "paper": path},
                )

            # Relations
            for r in items.get("relations", []):
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

    # ---------------------------
    # Public API
    # ---------------------------
    def run(self) -> None:
        self.ensure_schema()
        files = self.list_input_texts()
        for path in files:
            try:
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
                self.upsert_document(path=path, title=title, items=items)
                logger.info(f"Indexed: {path}")
            except Exception as e:
                logger.error(f"Failed indexing {path}: {e}")


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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    indexer = GraphIndexer(input_dirs=args.dirs, limit_files=args.limit)
    if args.init_schema_only:
        indexer.ensure_schema()
        logger.info("Schema initialized. Exiting as requested.")
        return
    indexer.run()


if __name__ == "__main__":
    main()


