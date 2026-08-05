import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings

class VectorStoreService:
    """
    Manages local ChromaDB vector database persistence for medical study document chunks.
    """

    def __init__(self, persist_dir: str = settings.VECTOR_DB_DIR, collection_name: str = "medical_study_docs"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 250):
        """
        Inserts document chunks and precomputed embeddings into ChromaDB in batches.
        """
        if not chunks or not embeddings:
            return

        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [c["metadata"] for c in batch_chunks]

            self.collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def search_similarity(self, query_embedding: List[float], top_k: int = 5, where_filter: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection for top_k most similar document chunks.
        """
        if not query_embedding:
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        matched_chunks = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx]
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx] if results["distances"] else 0.0
                
                # Convert cosine distance to similarity score
                similarity_score = float(1.0 - distance)

                matched_chunks.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": metadata,
                    "score": similarity_score
                })

        return matched_chunks

    def count(self) -> int:
        """Returns total number of indexed document chunks."""
        return self.collection.count()

    def clear(self):
        """Clears all indexed chunks from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

# Backward-compatible alias
VectorStoreManager = VectorStoreService

