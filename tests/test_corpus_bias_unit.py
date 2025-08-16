import os
from langchain.schema import Document

from retrieval.corpus_bias import rerank_with_primary_weight, apply_primary_corpus_bias


def make_doc(src: str, text: str = "t"):
    return Document(page_content=text, metadata={"source": src})


def test_boost_moves_primary_up():
    docs = [
        make_doc("papers_np/a.txt"),
        make_doc("papers/b.txt"),
        make_doc("papers_np/c.txt"),
        make_doc("papers/d.txt"),
    ]
    out = rerank_with_primary_weight(docs, primary_prefix="papers/", primary_boost=2.0, min_primary_share=0.0, final_k=4)
    sources = [d.metadata.get("source") for d in out]
    assert sources.index("papers/b.txt") < sources.index("papers_np/c.txt")
    assert sources.index("papers/d.txt") < sources.index("papers_np/a.txt")


def test_min_share_enforced_when_available():
    docs = [
        make_doc("papers_np/1.txt"),
        make_doc("papers/2.txt"),
        make_doc("papers_np/3.txt"),
        make_doc("papers/4.txt"),
        make_doc("papers/5.txt"),
        make_doc("papers_np/6.txt"),
    ]
    out = rerank_with_primary_weight(docs, primary_prefix="papers/", primary_boost=2.0, min_primary_share=0.6, final_k=5)
    top = out[:5]
    primary_cnt = sum(1 for d in top if d.metadata.get("source", "").startswith("papers/"))
    assert primary_cnt >= 3


def test_no_primary_available_keeps_order():
    docs = [make_doc("papers_np/a.txt"), make_doc("papers_np/b.txt"), make_doc("papers_np/c.txt")]
    out = rerank_with_primary_weight(docs, primary_prefix="papers/", primary_boost=2.0, min_primary_share=0.8, final_k=3)
    assert [d.metadata["source"] for d in out] == ["papers_np/a.txt", "papers_np/b.txt", "papers_np/c.txt"]


def test_stable_order_within_same_group():
    docs = [make_doc("papers/a.txt"), make_doc("papers/b.txt"), make_doc("papers_np/c.txt"), make_doc("papers_np/d.txt")]
    out = rerank_with_primary_weight(docs, primary_prefix="papers/", primary_boost=2.0, min_primary_share=0.0, final_k=4)
    sources = [d.metadata.get("source") for d in out]
    assert sources.index("papers/a.txt") < sources.index("papers/b.txt")
    assert sources.index("papers_np/c.txt") < sources.index("papers_np/d.txt")


def test_env_config_parsing_enabled():
    os.environ["PRIMARY_AUTHOR_BIAS_ENABLED"] = "true"
    os.environ["PRIMARY_AUTHOR_PREFIX"] = "papers/"
    os.environ["PRIMARY_AUTHOR_BOOST"] = "2.0"
    os.environ["PRIMARY_AUTHOR_MIN_SHARE"] = "0.8"
    os.environ["PRIMARY_AUTHOR_FINAL_K"] = "3"

    docs = [make_doc("papers_np/a.txt"), make_doc("papers/b.txt"), make_doc("papers_np/c.txt")]
    out = apply_primary_corpus_bias(docs)
    assert len(out) == 3
    assert out[0].metadata.get("source").startswith("papers/")


def test_recency_within_group_primary_first():
    a = make_doc("papers/a.txt"); a.metadata["year"] = 2018
    b = make_doc("papers/b.txt"); b.metadata["year"] = 2022
    c = make_doc("papers_np/c.txt"); c.metadata["year"] = 2023
    out = rerank_with_primary_weight([a, b, c], final_k=3, recency_enabled=True)
    sources = [d.metadata.get("source") for d in out]
    assert sources[0] == "papers/b.txt"


def test_recency_within_group_non_primary():
    a = make_doc("papers/a.txt"); a.metadata["year"] = 2019
    x = make_doc("papers_np/x.txt"); x.metadata["year"] = 2015
    y = make_doc("papers_np/y.txt"); y.metadata["year"] = 2021
    out = rerank_with_primary_weight([x, y, a], final_k=3, min_primary_share=1/3, recency_enabled=True)
    sources = [d.metadata.get("source") for d in out]
    assert sources[0] == "papers/a.txt"
    assert sources[1] == "papers_np/y.txt"
    assert sources[2] == "papers_np/x.txt"


def test_missing_year_preserves_order():
    p1 = make_doc("papers/p1.txt")
    p2 = make_doc("papers/p2.txt")
    np1 = make_doc("papers_np/np1.txt")
    out = rerank_with_primary_weight([np1, p2, p1], final_k=3, recency_enabled=True)
    sources = [d.metadata.get("source") for d in out]
    assert sources[:2] == ["papers/p2.txt", "papers/p1.txt"]


