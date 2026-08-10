from pathlib import Path
import tempfile
import fitz
import pytest

from app.api.routes.ingest import process_document_ingestion
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_retriever import HybridRetriever
from app.services.rag_chain import RAGChain
from app.services.vector_store import VectorStoreService


@pytest.fixture
def temp_medical_pdf():
    temp_dir = tempfile.mkdtemp()
    pdf_path = Path(temp_dir) / "synthetic_neuro_notes.pdf"

    doc = fitz.open()
    page = doc.new_page()

    text = (
        "CHAPTER 1: HYPERPYREXIA MANAGEMENT\n"
        "Hyperpyrexia is defined as a fever greater than 41.5 degrees Celsius. "
        "First-line treatment includes active core cooling and intravenous paracetamol administration. "
        "In severe cases, dantrolene sodium may be administered if neuroleptic malignant syndrome is suspected. "
        "Frequent neurological assessment using the Glasgow Coma Scale is vital."
    )
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), text)
    doc.save(str(pdf_path))
    doc.close()

    yield str(pdf_path)


def test_end_to_end_rag_flow(temp_medical_pdf):
    # 1. Ingest document via canonical pipeline
    ingest_res = process_document_ingestion(temp_medical_pdf)
    assert ingest_res["status"] == "success"
    assert ingest_res["parent_chunks_indexed"] > 0
    assert ingest_res["child_chunks_indexed"] > 0

    vector_store = VectorStoreService()
    embedder = EmbeddingService()
    retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)

    # 2. Issue medical query with relevant context
    contexts = retriever.retrieve("How is hyperpyrexia treated?")
    assert len(contexts) > 0
    assert "paracetamol" in contexts[0]["text"].lower() or "dantrolene" in contexts[0]["text"].lower()
    assert contexts[0]["metadata"]["source"] == Path(temp_medical_pdf).name

    # 3. Test evidence gate refusal on completely unrelated question
    unrelated_contexts = retriever.retrieve("What is quantum electrodynamics in astrophysics?")
    assert len(unrelated_contexts) == 0

    rag_chain = RAGChain()
    refusal_response = rag_chain.generate_response(
        "What is quantum electrodynamics in astrophysics?", unrelated_contexts
    )
    assert refusal_response["grounded"] is False
    assert refusal_response["sources"] == []
    assert "could not find sufficient evidence" in refusal_response["answer"].lower()
