from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.ingestion.embedder import EmbeddingService
from app.services.vector_store import VectorStoreManager
from app.services.hybrid_retriever import HybridRetriever
from app.services.rag_chain import RAGChain

router = APIRouter(prefix="/api/query", tags=["Medical Query"])

class QueryRequest(BaseModel):
    query: str

class SourceItem(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    heading: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    grounded: bool
    retrieved_contexts: Optional[List[Dict[str, Any]]] = None

embedder = EmbeddingService()
vector_store = VectorStoreManager()
hybrid_retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
rag_chain = RAGChain()

@router.post("", response_model=QueryResponse)
def query_medical_assistant(req: QueryRequest):
    """Submits a medical query to the Hybrid RAG pipeline."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # 1. Hybrid Retrieval (Vector + BM25 + RRF + Rerank + Parent Context)
    contexts = hybrid_retriever.retrieve(req.query)

    # 2. RAG LLM Context Synthesis & Citation Generation
    result = rag_chain.generate_response(req.query, contexts)

    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        sources=result["sources"],
        grounded=result["grounded"],
        retrieved_contexts=contexts
    )
