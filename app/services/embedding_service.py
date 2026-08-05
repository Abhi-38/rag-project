from sentence_transformers import SentenceTransformer
from typing import List
from app.core.config import settings

class EmbeddingService:
    """
    Embedding service wrapping HuggingFace / BAAI models (default: BAAI/bge-small-en-v1.5).
    Produces normalized vector representations for dense semantic search.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encodes a list of chunk texts into normalized embedding vectors with progress reporting."""
        if not texts:
            return []
        print(f"Generating embeddings for {len(texts)} semantic chunks using {self.model_name}...")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True, batch_size=32)
        print("Embedding generation completed successfully.")
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encodes a single user search query into a normalized vector."""
        if not query.strip():
            return []
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
