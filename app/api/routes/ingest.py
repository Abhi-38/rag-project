import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestion.loaders.pipeline import IngestionPipeline
from app.services.embedding_service import EmbeddingService
from app.services.progress_tracker import progress_tracker
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])


class IngestRequest(BaseModel):
    file_path: str


# Singleton service instances
embedder = EmbeddingService()
vector_store = VectorStoreService()
pipeline = IngestionPipeline()


def process_document_ingestion(doc_path: str) -> Dict[str, Any]:
    """Canonical document ingestion pipeline function with real-time progress, percentage, and ETA reporting."""
    path = Path(doc_path)
    if not path.exists():
        msg = f"File not found at {doc_path}"
        progress_tracker.error(msg)
        return {"status": "error", "message": msg}

    try:
        # Start Progress Tracking
        progress_tracker.start(path.name)

        # 1. Parse & Parent-Child Chunking
        progress_tracker.update_stage("Parsing document into semantic parent-child sections...", 1, 15.0)
        chunks_dict = pipeline.process_file(str(path))
        parents = chunks_dict.get("parents", [])
        children = chunks_dict.get("children", [])

        if not children:
            msg = "No readable text extracted from document."
            progress_tracker.error(msg)
            return {"status": "error", "message": msg}

        # 2. Store Parent Chunks in Persistent Parent Store
        progress_tracker.update_stage("Storing parent contexts in database...", 1, 20.0)
        vector_store.add_parent_chunks(parents)

        # 3. Generate Dense Embeddings for Child Chunks (with batch progress & ETA)
        child_texts = [c["text"] for c in children]
        embeddings = embedder.embed_documents(
            child_texts, progress_callback=progress_tracker.update_embedding_progress
        )

        # 4. Store Child Chunks & Embeddings in Vector Database
        progress_tracker.update_stage("Indexing vectors into ChromaDB...", 3, 95.0)
        vector_store.add_child_chunks(children=children, embeddings=embeddings)

        # Complete Progress Tracking
        progress_tracker.complete(len(parents), len(children))

        return {
            "status": "success",
            "file": path.name,
            "parent_chunks_indexed": len(parents),
            "child_chunks_indexed": len(children),
            "vector_db_child_count": vector_store.count(),
        }
    except Exception as e:
        progress_tracker.error(str(e))
        return {"status": "error", "message": str(e)}


@router.get("/stats")
def get_ingestion_stats():
    """Returns total count of indexed document chunks in ChromaDB."""
    return {
        "status": "online",
        "total_child_chunks": vector_store.count(),
    }


@router.get("/progress")
def get_ingestion_progress():
    """Returns current real-time ingestion stage, percentage, and ETA."""
    return progress_tracker.get_progress()


@router.post("")
def ingest_document(req: IngestRequest, background_tasks: BackgroundTasks):
    """Triggers document ingestion pipeline in background so event loop remains unblocked for progress polling."""
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"File not found at {req.file_path}")

    background_tasks.add_task(process_document_ingestion, str(path))
    return {
        "status": "processing",
        "file": path.name,
        "message": f"Document ingestion started in background for {path.name}.",
    }


@router.post("/upload")
async def upload_and_ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a document and triggers ingestion in background so progress polling updates in real-time."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file format '{suffix}'. Only .pdf and .docx are supported."
        )

    save_dir = Path("./data/sample_docs")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file.filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(process_document_ingestion, str(file_path))
    return {
        "status": "processing",
        "file": file.filename,
        "message": f"Document ingestion started in background for {file.filename}.",
    }
