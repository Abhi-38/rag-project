import pytest
import shutil
import tempfile
from pathlib import Path
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

@pytest.fixture
def temp_vector_store():
    temp_dir = tempfile.mkdtemp()
    store = VectorStoreService(persist_dir=temp_dir, collection_name="test_collection")
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_embedding_service():
    embedder = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    texts = ["Trigeminal neuralgia pain treatment.", "Glioblastoma multiforme tumor diagnosis."]
    embeddings = embedder.embed_documents(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # bge-small-en-v1.5 output dimension is 384

    query_emb = embedder.embed_query("trigeminal neuralgia")
    assert len(query_emb) == 384

def test_vector_store_add_and_search(temp_vector_store):
    embedder = EmbeddingService(model_name="BAAI/bge-small-en-v1.5")
    
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Carbamazepine is the primary medical treatment for trigeminal neuralgia.",
            "metadata": {"source": "pharma.pdf", "page": 10, "chunk_id": "chunk_1"}
        },
        {
            "chunk_id": "chunk_2",
            "text": "Subarachnoid hemorrhage presents with sudden thunderstorm headache.",
            "metadata": {"source": "emergency.pdf", "page": 5, "chunk_id": "chunk_2"}
        }
    ]
    
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)
    
    temp_vector_store.add_chunks(chunks, embeddings)
    assert temp_vector_store.count() == 2

    query_vector = embedder.embed_query("What is used to treat trigeminal neuralgia?")
    results = temp_vector_store.search_similarity(query_vector, top_k=1)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["metadata"]["source"] == "pharma.pdf"
    assert results[0]["metadata"]["page"] == 10
