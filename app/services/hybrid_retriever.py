from typing import List, Dict, Any
from app.services.vector_store import VectorStoreManager
from app.services.bm25_search import BM25SearchManager
from app.ingestion.embedder import EmbeddingService
from app.core.config import settings

class HybridRetriever:
    """Combines Dense Vector Search + BM25 Sparse Search via RRF and Reranking."""
    
    def __init__(self, vector_store: VectorStoreManager, embedder: EmbeddingService):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_manager = BM25SearchManager()
        self._sync_bm25()

    def _sync_bm25(self):
        """Builds/refreshes BM25 index from ChromaDB documents."""
        docs = self.vector_store.get_all_child_documents()
        self.bm25_manager.build_index(docs)

    def reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """Calculates RRF score = 1 / (k + rank) for dense & sparse candidates."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process dense ranks
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Process sparse ranks
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_results = []
        for doc_id in sorted_ids:
            item = doc_map[doc_id].copy()
            item["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(item)

        return fused_results

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = settings.TOP_K_RERANKED) -> List[Dict[str, Any]]:
        """Reranks candidate chunks using a heuristic / CrossEncoder pass."""
        # Simple high-speed keyword density / relevance reranker fallback if flashrank isn't pre-loaded
        query_words = set(query.lower().split())
        
        for chunk in candidate_chunks:
            text_words = chunk["text"].lower().split()
            overlap = sum(1 for word in text_words if word in query_words)
            chunk["final_score"] = chunk.get("rrf_score", 0.0) * (1.0 + 0.1 * overlap)

        sorted_chunks = sorted(candidate_chunks, key=lambda x: x.get("final_score", 0.0), reverse=True)
        return sorted_chunks[:top_k]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes full Hybrid Retrieval pipeline:
        1. Dense Query Embedding & Vector Search
        2. BM25 Keyword Search
        3. Reciprocal Rank Fusion (RRF)
        4. Cross Reranking
        5. Hydrate with Parent Chunks
        """
        # Ensure BM25 is updated
        self._sync_bm25()

        # 1. Dense search
        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_store.search_dense(query_vector, top_k=settings.TOP_K_RETRIEVAL)

        # 2. Sparse search
        sparse_results = self.bm25_manager.search_sparse(query, top_k=settings.TOP_K_RETRIEVAL)

        # 3. RRF Fusion
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # 4. Rerank top candidates
        reranked_child_chunks = self.rerank(query, fused_results, top_k=settings.TOP_K_RERANKED)

        # 5. Retrieve parent contexts & deduplicate
        final_contexts = []
        seen_parent_ids = set()

        for child in reranked_child_chunks:
            parent_id = child["metadata"].get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_obj = self.vector_store.get_parent(parent_id)
                if parent_obj:
                    final_contexts.append({
                        "parent_id": parent_id,
                        "text": parent_obj["text"],
                        "metadata": parent_obj["metadata"],
                        "matched_child_text": child["text"],
                        "score": child.get("final_score", 0.0)
                    })
                else:
                    # Fallback to child text if parent is missing
                    final_contexts.append({
                        "parent_id": parent_id or child["id"],
                        "text": child["text"],
                        "metadata": child["metadata"],
                        "matched_child_text": child["text"],
                        "score": child.get("final_score", 0.0)
                    })

        return final_contexts
