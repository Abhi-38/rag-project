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
pipeline = IngestionPipeline(chunk_size=800, chunk_overlap=150)

def process_pdf_ingestion(pdf_path: str) -> Dict[str, Any]:
    """Ingestion pipeline function: Parses, chunks, embeds, and stores in ChromaDB."""
    path = Path(pdf_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found at {pdf_path}"}

    # 1. Parse & Chunk
    chunks = pipeline.process_file(str(path))
    if not chunks:
        return {"status": "error", "message": "No readable text extracted from document."}

    # 2. Generate Embeddings
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(chunk_texts)

    # 3. Store in ChromaDB Vector Database
    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    return {
        "status": "success",
        "file": path.name,
        "total_chunks_indexed": len(chunks),
        "vector_db_total_count": vector_store.count()
    }

@router.post("")
def ingest_document(req: IngestRequest):
    """Triggers document ingestion pipeline."""
    res = process_pdf_ingestion(req.file_path)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res
