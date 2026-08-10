from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any

from app.ingestion.loaders.pipeline import IngestionPipeline
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])

class IngestRequest(BaseModel):
    file_path: str

# Singleton service instances
embedder = EmbeddingService()
vector_store = VectorStoreService()
pipeline = IngestionPipeline()

def process_document_ingestion(doc_path: str) -> Dict[str, Any]:
    """Canonical document ingestion pipeline function: Parses, chunks (Parent-Child), embeds, and stores in ChromaDB."""
    path = Path(doc_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found at {doc_path}"}

    # 1. Parse & Parent-Child Chunking
    chunks_dict = pipeline.process_file(str(path))
    parents = chunks_dict.get("parents", [])
    children = chunks_dict.get("children", [])

    if not children:
        return {"status": "error", "message": "No readable text extracted from document."}

    # 2. Store Parent Chunks in Persistent Parent Store
    vector_store.add_parent_chunks(parents)

    # 3. Generate Dense Embeddings for Child Chunks
    child_texts = [c["text"] for c in children]
    embeddings = embedder.embed_documents(child_texts)

    # 4. Store Child Chunks & Embeddings in Vector Database
    vector_store.add_child_chunks(children=children, embeddings=embeddings)

    return {
        "status": "success",
        "file": path.name,
        "parent_chunks_indexed": len(parents),
        "child_chunks_indexed": len(children),
        "vector_db_child_count": vector_store.count()
    }

@router.post("")
def ingest_document(req: IngestRequest):
    """Triggers document ingestion pipeline."""
    res = process_document_ingestion(req.file_path)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res
