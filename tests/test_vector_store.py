import shutil
import tempfile
import pytest

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService


@pytest.fixture
def temp_vector_store():
    temp_dir = tempfile.mkdtemp()
    store = VectorStoreService(
        persist_dir=temp_dir,
        child_collection_name="test_children",
        parent_collection_name="test_parents",
    )
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedding_service():
    embedder = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    texts = ["Trigeminal neuralgia pain treatment.", "Glioblastoma multiforme tumor diagnosis."]
    embeddings = embedder.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384

    query_emb = embedder.embed_query("trigeminal neuralgia")
    assert len(query_emb) == 384


def test_vector_store_parent_child_indexing(temp_vector_store):
    embedder = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")

    parents = [
        {
            "parent_id": "p_1",
            "text": "Full paragraph: Carbamazepine is the primary medical treatment for trigeminal neuralgia.",
            "metadata": {"source": "pharma.pdf", "page": 10, "type": "parent"},
        }
    ]

    children = [
        {
            "child_id": "c_1",
            "parent_id": "p_1",
            "text": "Carbamazepine treats trigeminal neuralgia.",
            "raw_text": "Carbamazepine treats trigeminal neuralgia.",
            "metadata": {
                "child_id": "c_1",
                "parent_id": "p_1",
                "source": "pharma.pdf",
                "page": 10,
                "type": "child",
            },
        }
    ]

    embeddings = embedder.embed_documents([c["text"] for c in children])

    temp_vector_store.add_parent_chunks(parents)
    temp_vector_store.add_child_chunks(children, embeddings)

    assert temp_vector_store.count() == 1

    # Dense Search
    query_vector = embedder.embed_query("What is used to treat trigeminal neuralgia?")
    results = temp_vector_store.search_dense(query_vector, top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "c_1"
    assert results[0]["metadata"]["parent_id"] == "p_1"

    # Parent Retrieval
    parent_obj = temp_vector_store.get_parent("p_1")
    assert parent_obj is not None
    assert parent_obj["parent_id"] == "p_1"
    assert "Carbamazepine is the primary medical treatment" in parent_obj["text"]

    # All Child Documents Retrieval for BM25
    all_children = temp_vector_store.get_all_child_documents()
    assert len(all_children) == 1
    assert all_children[0]["id"] == "c_1"
