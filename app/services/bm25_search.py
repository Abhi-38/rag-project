import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25SearchManager:
    """Manages sparse BM25 keyword index for exact medical term matching."""
    
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.documents: List[Dict[str, Any]] = []

    def tokenize(self, text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenization."""
        return re.findall(r'\w+', text.lower())

    def build_index(self, documents: List[Dict[str, Any]]):
        """Builds BM25 index from document list."""
        self.documents = documents
        if not documents:
            self.bm25 = None
            return

        corpus_tokenized = [self.tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(corpus_tokenized)

    def search_sparse(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Executes BM25 keyword search on tokenized query."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort documents by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                results.append(doc)

        return results
