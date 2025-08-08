import os
import glob
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def list_txt_files(input_dirs: Optional[List[str]] = None, limit: Optional[int] = None) -> List[str]:
    input_dirs = input_dirs or ["papers", "papers_np"]
    files: List[str] = []
    for d in input_dirs:
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))
    if limit:
        files = files[:limit]
    return files


class GraphInspector:
    def __init__(self) -> None:
        self.uri = _get_env("NEO4J_URI")
        self.user = _get_env("NEO4J_USER")
        self.password = _get_env("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def inspect(self, paper_paths: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        with self.driver.session() as session:
            # Papers
            papers = session.run(
                "MATCH (p:Paper) WHERE p.path IN $paths RETURN p.path AS path, p.title AS title ORDER BY path",
                {"paths": paper_paths},
            ).data()
            result["papers"] = papers

            # Entities mentioning these papers
            entities = session.run(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(p:Paper)
                WHERE p.path IN $paths
                RETURN DISTINCT e.name AS name, e.type AS type
                ORDER BY name
                """,
                {"paths": paper_paths},
            ).data()
            result["entities"] = entities

            # Claims supported by these papers
            claims = session.run(
                """
                MATCH (c:Claim)-[:SUPPORTED_BY]->(p:Paper)
                WHERE p.path IN $paths
                RETURN DISTINCT c.id AS id, c.text AS text, c.confidence AS confidence
                ORDER BY id
                """,
                {"paths": paper_paths},
            ).data()
            result["claims"] = claims

            # Edges: MENTIONED_IN
            mentioned = session.run(
                """
                MATCH (e:Entity)-[m:MENTIONED_IN]->(p:Paper)
                WHERE p.path IN $paths
                RETURN e.name AS entity, 'MENTIONED_IN' AS rel, p.path AS paper
                ORDER BY entity, paper
                """,
                {"paths": paper_paths},
            ).data()
            result["mentioned_in"] = mentioned

            # Edges: SUPPORTED_BY
            supported = session.run(
                """
                MATCH (c:Claim)-[r:SUPPORTED_BY]->(p:Paper)
                WHERE p.path IN $paths
                RETURN c.id AS claim, 'SUPPORTED_BY' AS rel, p.path AS paper
                ORDER BY claim, paper
                """,
                {"paths": paper_paths},
            ).data()
            result["supported_by"] = supported

            # Edges: RELATES_TO among entities connected to selected papers
            relates = session.run(
                """
                MATCH (e1:Entity)-[:MENTIONED_IN]->(p:Paper)
                WHERE p.path IN $paths
                WITH DISTINCT e1
                MATCH (e1)-[rel:RELATES_TO]-(e2:Entity)
                RETURN e1.name AS source, 'RELATES_TO' AS rel, rel.relation_type AS relation_type, e2.name AS target
                ORDER BY source, target
                """,
                {"paths": paper_paths},
            ).data()
            result["relates_to"] = relates

        return result


def _parse_args(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description="Inspect graph nodes/edges for selected papers")
    parser.add_argument("--papers", nargs="*", help="Explicit list of paper .txt paths to inspect")
    parser.add_argument("--limit", type=int, default=3, help="Use first-N .txt files if --papers not provided")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    papers = args.papers if args.papers else list_txt_files(limit=args.limit)
    inspector = GraphInspector()
    data = inspector.inspect(papers)

    print("PAPERS:")
    for p in data["papers"]:
        print(f"- {p['path']} | title={p.get('title')}")

    print("\nENTITIES:")
    for e in data["entities"]:
        print(f"- {e['name']} | type={e.get('type')}")

    print("\nCLAIMS:")
    for c in data["claims"]:
        print(f"- {c['id']} | {c.get('text')} | conf={c.get('confidence')}")

    print("\nEDGES: MENTIONED_IN")
    for m in data["mentioned_in"]:
        print(f"- ({m['entity']}) -[{m['rel']}]-> ({m['paper']})")

    print("\nEDGES: SUPPORTED_BY")
    for s in data["supported_by"]:
        print(f"- ({s['claim']}) -[{s['rel']}]-> ({s['paper']})")

    print("\nEDGES: RELATES_TO")
    for r in data["relates_to"]:
        print(f"- ({r['source']}) -[{r['rel']}:{r.get('relation_type')}] -> ({r['target']})")


if __name__ == "__main__":
    main()


