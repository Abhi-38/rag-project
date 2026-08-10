from typing import List, Dict, Any
from app.services.vector_store import VectorStoreManager
from app.services.bm25_search import BM25SearchManager
from app.services.embedding_service import EmbeddingService
from app.core.config import settings

class HybridRetriever:
    """Combines Dense Vector Search + BM25 Sparse Search via RRF, Reranking, Parent Context Hydration, and Evidence Gating."""

    def __init__(self, vector_store: VectorStoreManager, embedder: EmbeddingService):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_manager = BM25SearchManager()
        self._sync_bm25()

    def _sync_bm25(self):
        """Builds/refreshes BM25 index from persistent ChromaDB child documents."""
        docs = self.vector_store.get_all_child_documents()
        self.bm25_manager.build_index(docs)

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
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

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for doc_id in sorted_ids:
            item = doc_map[doc_id].copy()
            item["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(item)

        return fused_results

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = settings.TOP_K_RERANKED
    ) -> List[Dict[str, Any]]:
        """Lightweight relevance reranker balancing RRF score and lexical term density."""
        if not candidate_chunks:
            return []

        query_words = set(w for w in query.lower().split() if len(w) > 2)

        for chunk in candidate_chunks:
            text_words = set(chunk.get("text", "").lower().split())
            overlap = sum(1 for word in query_words if word in text_words)
            dense_score = chunk.get("score", 0.0)

            # If there is zero term overlap AND dense vector similarity is low (< 0.40), set final_score to 0.0
            if overlap == 0 and dense_score < 0.40:
                chunk["final_score"] = 0.0
            else:
                chunk["final_score"] = chunk.get("rrf_score", 0.0) * (1.0 + 0.2 * overlap)

        sorted_chunks = sorted(candidate_chunks, key=lambda x: x.get("final_score", 0.0), reverse=True)
        return sorted_chunks[:top_k]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes full Hybrid Retrieval pipeline:
        1. Refresh BM25 index from persistent store
        2. Dense Vector Search (ChromaDB)
        3. Sparse Keyword Search (BM25)
        4. Reciprocal Rank Fusion (RRF)
        5. Reranking Pass
        6. Evidence Sufficiency Gate
        7. Parent Context Hydration
        """
        # 1. Sync BM25 index
        self._sync_bm25()

        # 2. Dense search
        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_store.search_dense(query_vector, top_k=settings.TOP_K_RETRIEVAL)

        # 3. Sparse search
        sparse_results = self.bm25_manager.search_sparse(query, top_k=settings.TOP_K_RETRIEVAL)

        # 4. RRF Fusion
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # 5. Rerank top candidates
        reranked_child_chunks = self.rerank(query, fused_results, top_k=settings.TOP_K_RERANKED)

        # 6. Evidence Sufficiency Gate
        if not reranked_child_chunks:
            return []

        top_score = reranked_child_chunks[0].get("final_score", 0.0)
        if top_score < settings.EVIDENCE_THRESHOLD:
            # Insufficient retrieval evidence
            return []

        # 7. Parent Context Hydration & Deduplication
        final_contexts = []
        seen_parent_ids = set()

        for child in reranked_child_chunks:
            metadata = child.get("metadata", {})
            parent_id = metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_obj = self.vector_store.get_parent(parent_id)
                if parent_obj:
                    final_contexts.append({
                        "parent_id": parent_id,
                        "text": parent_obj["text"],
                        "metadata": parent_obj["metadata"],
                        "matched_child_text": child.get("text"),
                        "score": child.get("final_score", 0.0)
                    })
                else:
                    final_contexts.append({
                        "parent_id": parent_id or child.get("id"),
                        "text": child.get("text"),
                        "metadata": metadata,
                        "matched_child_text": child.get("text"),
                        "score": child.get("final_score", 0.0)
                    })

        return final_contexts
