from typing import Callable, List, Optional
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    """Embedding service wrapping HuggingFace / BAAI models (default: BAAI/bge-m3)."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def embed_documents(
        self, texts: List[str], progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        """Encodes chunk texts into normalized vectors with optional real-time progress callbacks."""
        if not texts:
            return []

        batch_size = 32
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = self.model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(batch_emb.tolist())

            processed = min(i + batch_size, total)
            if progress_callback:
                progress_callback(processed, total)

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Encodes a single search query into a normalized vector."""
        if not query.strip():
            return []
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
