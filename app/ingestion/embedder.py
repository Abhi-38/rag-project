from sentence_transformers import SentenceTransformer
from typing import List
from app.core.config import settings

class EmbeddingService:
    """Wrapper around SentenceTransformer / BGE embedding models."""
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        # Falls back gracefully if bge-m3 download is large or slow on limited devices
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            fallback_model = "BAAI/bge-small-en-v1.5"
            print(f"Warning: Failed to load {model_name}. Falling back to {fallback_model}.")
            self.model = SentenceTransformer(fallback_model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Encodes a list of texts into dense vector representations."""
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encodes a single search query."""
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
