import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings

class VectorStoreService:
    """
    Manages persistent ChromaDB vector database with dual collections:
    - medical_child_chunks: Child chunk embeddings for dense vector retrieval.
    - medical_parent_chunks: Parent chunk documents for context hydration.
    """

    def __init__(
        self,
        persist_dir: str = settings.VECTOR_DB_DIR,
        child_collection_name: str = "medical_child_chunks",
        parent_collection_name: str = "medical_parent_chunks"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.child_collection = self.client.get_or_create_collection(
            name=child_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_collection = self.client.get_or_create_collection(
            name=parent_collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_parent_chunks(self, parents: List[Dict[str, Any]], batch_size: int = 250):
        """Stores parent document chunks in persistent parent collection."""
        if not parents:
            return

        total_parents = len(parents)
        for i in range(0, total_parents, batch_size):
            batch = parents[i:i + batch_size]
            ids = [p["parent_id"] for p in batch]
            documents = [p["text"] for p in batch]
            metadatas = [p["metadata"] for p in batch]
            dummy_embeddings = [[0.0] * 1024 for _ in batch]

            self.parent_collection.upsert(
                ids=ids,
                embeddings=dummy_embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def add_child_chunks(self, children: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 250):
        """Inserts child chunks and precomputed embeddings into ChromaDB in batches."""
        if not children or not embeddings:
            return

        total_children = len(children)
        for i in range(0, total_children, batch_size):
            batch_children = children[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [c["child_id"] for c in batch_children]
            documents = [c["text"] for c in batch_children]
            metadatas = [c["metadata"] for c in batch_children]

            self.child_collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = settings.TOP_K_RETRIEVAL,
        where_filter: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """Queries child collection for top_k most similar document chunks."""
        if not query_embedding:
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.child_collection.query(**kwargs)

        matched_chunks = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx]
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx] if results["distances"] else 0.0
                similarity_score = float(1.0 - distance)

                matched_chunks.append({
                    "id": chunk_id,
                    "text": doc_text,
                    "metadata": metadata,
                    "score": similarity_score
                })

        return matched_chunks

    def get_parent(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a parent document and metadata by parent_id."""
        if not parent_id:
            return None
        res = self.parent_collection.get(ids=[parent_id], include=["documents", "metadatas"])
        if res and res["ids"] and len(res["ids"]) > 0:
            return {
                "parent_id": res["ids"][0],
                "text": res["documents"][0],
                "metadata": res["metadatas"][0]
            }
        return None

    def get_all_child_documents(self) -> List[Dict[str, Any]]:
        """Retrieves all indexed child documents for building/refreshing the BM25 index."""
        res = self.child_collection.get(include=["documents", "metadatas"])
        docs = []
        if res and res["ids"]:
            for idx in range(len(res["ids"])):
                docs.append({
                    "id": res["ids"][idx],
                    "text": res["documents"][idx],
                    "metadata": res["metadatas"][idx]
                })
        return docs

    def count(self) -> int:
        """Returns total number of indexed child document chunks."""
        return self.child_collection.count()

    def clear(self):
        """Clears all indexed chunks from both child and parent collections."""
        child_name = self.child_collection.name
        parent_name = self.parent_collection.name
        self.client.delete_collection(child_name)
        self.client.delete_collection(parent_name)

        self.child_collection = self.client.get_or_create_collection(
            name=child_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_collection = self.client.get_or_create_collection(
            name=parent_name,
            metadata={"hnsw:space": "cosine"}
        )

# Backward-compatible alias
VectorStoreManager = VectorStoreService
