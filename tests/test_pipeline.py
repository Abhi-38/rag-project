import pytest
from app.ingestion.chunking.chunker import ParentChildChunker
from app.services.bm25_search import BM25SearchManager
from app.services.hybrid_retriever import HybridRetriever


def test_parent_child_chunker():
    docs = [
        {
            "text": "Trigeminal neuralgia is a chronic pain condition that affects the trigeminal nerve. " * 30,
            "metadata": {
                "source": "test_book.pdf",
                "page": 1,
                "heading": "NEUROSURGERY PRINCIPLES",
                "file_type": "pdf",
            },
        }
    ]

    chunker = ParentChildChunker(parent_size=500, child_size=100)
    result = chunker.create_chunks(docs)

    assert "parents" in result
    assert "children" in result
    assert len(result["parents"]) > 0
    assert len(result["children"]) > 0

    child = result["children"][0]
    assert "parent_id" in child
    assert any(p["parent_id"] == child["parent_id"] for p in result["parents"])


def test_bm25_search():
    docs = [
        {"id": "doc1", "text": "Carbamazepine is first-line treatment for trigeminal neuralgia."},
        {"id": "doc2", "text": "Subarachnoid hemorrhage causes sudden severe headache."},
        {"id": "doc3", "text": "Glioblastoma multiforme is an aggressive brain tumor."},
    ]

    manager = BM25SearchManager()
    manager.build_index(docs)

    results = manager.search_sparse("trigeminal neuralgia treatment")
    assert len(results) > 0
    assert results[0]["id"] == "doc1"


def test_reciprocal_rank_fusion():
    dense_results = [
        {"id": "docA", "text": "Text A"},
        {"id": "docB", "text": "Text B"},
    ]
    sparse_results = [
        {"id": "docB", "text": "Text B"},
        {"id": "docC", "text": "Text C"},
    ]

    retriever = HybridRetriever.__new__(HybridRetriever)
    fused = retriever.reciprocal_rank_fusion(dense_results, sparse_results, k=60)

    assert len(fused) == 3
    assert fused[0]["id"] == "docB"


def test_reranking_and_evidence_gate():
    candidates = [
        {"id": "doc1", "text": "Carbamazepine treats trigeminal neuralgia pain", "rrf_score": 0.03},
        {"id": "doc2", "text": "Subarachnoid hemorrhage headache", "rrf_score": 0.01},
    ]

    retriever = HybridRetriever.__new__(HybridRetriever)
    reranked = retriever.rerank("trigeminal neuralgia treatment", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["id"] == "doc1"
    assert reranked[0]["final_score"] > reranked[1]["final_score"]
