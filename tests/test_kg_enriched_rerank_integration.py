import os
from unittest.mock import Mock

from langchain.schema import Document

from retrieval.kg_enriched_retrieval import KGEnrichedRetriever


class DummyKGFilter:
    def __init__(self):
        pass
    def filter_and_format_kg_results(self, kg_dicts, query):
        return "context"


def make_doc(src: str, text: str = "t"):
    return Document(page_content=text, metadata={"source": src})


def test_rerank_applies_after_vector_search_with_recency():
    os.environ["PRIMARY_AUTHOR_BIAS_ENABLED"] = "true"
    os.environ["PRIMARY_AUTHOR_PREFIX"] = "papers/"
    os.environ["PRIMARY_AUTHOR_MIN_SHARE"] = "0.6"
    os.environ["PRIMARY_AUTHOR_FINAL_K"] = "5"

    graph_ret = Mock()
    graph_ret.get_relevant_documents.return_value = [make_doc("papers_np/kg.txt", "kg")]

    vec_ret = Mock()
    docs = [
        make_doc("papers_np/1.txt"),
        make_doc("papers/2.txt"),
        make_doc("papers_np/3.txt"),
        make_doc("papers/4.txt"),
        make_doc("papers/5.txt"),
        make_doc("papers_np/6.txt"),
    ]
    # Year metadata to test recency within groups
    for i, d in enumerate(docs):
        if d.metadata["source"].startswith("papers/"):
            d.metadata["year"] = 2022 + i
        else:
            d.metadata["year"] = 2018 + i
    vec_ret.get_relevant_documents.return_value = docs

    kg_filter = DummyKGFilter()

    retriever = KGEnrichedRetriever(graph_retriever=graph_ret, vector_retriever=vec_ret, kg_filter=kg_filter)
    out = retriever.get_relevant_documents("What is S8?")
    top = out[:5]
    primary_cnt = sum(1 for d in top if d.metadata.get("source", "").startswith("papers/"))
    assert primary_cnt >= 3


