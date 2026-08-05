import pytest
from app.ingestion.chunker import ParentChildChunker
from app.services.bm25_search import BM25SearchManager
from app.services.hybrid_retriever import HybridRetriever

def test_parent_child_chunker():
    pages_data = [
        {
            "page": 1,
            "source": "test_book.pdf",
            "heading": "NEUROSURGERY PRINCIPLES",
            "text": "Trigeminal neuralgia is a chronic pain condition that affects the trigeminal nerve. " * 30
        }
    ]
    
    chunker = ParentChildChunker(parent_size=500, child_size=100)
    result = chunker.create_chunks(pages_data)
    
    assert "parents" in result
    assert "children" in result
    assert len(result["parents"]) > 0
    assert len(result["children"]) > 0
    
    # Verify child links back to parent_id
    child = result["children"][0]
    assert "parent_id" in child
    assert child["parent_id"] in result["parents"]

def test_bm25_search():
    docs = [
        {"id": "doc1", "text": "Carbamazepine is first-line treatment for trigeminal neuralgia."},
        {"id": "doc2", "text": "Subarachnoid hemorrhage causes sudden severe headache."},
        {"id": "doc3", "text": "Glioblastoma multiforme is an aggressive brain tumor."}
    ]
    
    manager = BM25SearchManager()
    manager.build_index(docs)
    
    results = manager.search_sparse("trigeminal neuralgia treatment")
    assert len(results) > 0
    assert results[0]["id"] == "doc1"

def test_reciprocal_rank_fusion():
    dense_results = [
        {"id": "docA", "text": "Text A"},
        {"id": "docB", "text": "Text B"}
    ]
    sparse_results = [
        {"id": "docB", "text": "Text B"},
        {"id": "docC", "text": "Text C"}
    ]
    
    # Dummy mock retriever for testing fusion function
    retriever = HybridRetriever.__new__(HybridRetriever)
    fused = retriever.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    
    assert len(fused) == 3
    # docB was rank 1 in sparse and rank 2 in dense, so it should rank highest in fusion
    assert fused[0]["id"] == "docB"
