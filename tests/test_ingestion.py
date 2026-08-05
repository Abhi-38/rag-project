import pytest
from pathlib import Path
from app.ingestion.chunking.chunker import SemanticChunker
from app.ingestion.loaders.pipeline import IngestionPipeline

def test_semantic_chunker_metadata():
    docs = [
        {
            "text": "Neurosurgery requires precise understanding of cerebral anatomy and cerebrovascular disorders. " * 30,
            "metadata": {
                "source": "ramamurthi_neurosurgery.pdf",
                "page": 42,
                "file_type": "pdf"
            }
        }
    ]
    
    chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_documents(docs)
    
    assert len(chunks) > 0
    first_chunk = chunks[0]
    
    assert "chunk_id" in first_chunk
    assert "text" in first_chunk
    assert first_chunk["metadata"]["source"] == "ramamurthi_neurosurgery.pdf"
    assert first_chunk["metadata"]["page"] == 42
    assert "chunk_id" in first_chunk["metadata"]

def test_ingestion_pipeline_pdf_parsing():
    sample_pdf = Path("./data/sample_docs/Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not present in environment")

    pipeline = IngestionPipeline(chunk_size=800, chunk_overlap=150)
    chunks = pipeline.process_file(str(sample_pdf))

    assert len(chunks) > 0
    assert "chunk_id" in chunks[0]
    assert chunks[0]["metadata"]["source"] == sample_pdf.name
    assert chunks[0]["metadata"]["page"] >= 1
