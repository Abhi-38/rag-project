from pathlib import Path
import pytest
from app.ingestion.chunking.chunker import ParentChildChunker
from app.ingestion.loaders.pipeline import IngestionPipeline


def test_parent_child_chunker_metadata():
    docs = [
        {
            "text": "Neurosurgery requires precise understanding of cerebral anatomy and cerebrovascular disorders. " * 20,
            "metadata": {
                "source": "ramamurthi_neurosurgery.pdf",
                "page": 42,
                "heading": "Cerebral Anatomy",
                "file_type": "pdf",
            },
        }
    ]

    chunker = ParentChildChunker(parent_size=500, parent_overlap=50, child_size=100, child_overlap=20)
    result = chunker.create_chunks(docs)

    assert "parents" in result
    assert "children" in result
    assert len(result["parents"]) > 0
    assert len(result["children"]) > 0

    first_parent = result["parents"][0]
    assert "parent_id" in first_parent
    assert first_parent["metadata"]["source"] == "ramamurthi_neurosurgery.pdf"
    assert first_parent["metadata"]["page"] == 42
    assert first_parent["metadata"]["heading"] == "Cerebral Anatomy"

    first_child = result["children"][0]
    assert "child_id" in first_child
    assert "parent_id" in first_child
    assert first_child["parent_id"] == first_parent["parent_id"]
    assert first_child["metadata"]["page"] == 42


def test_ingestion_pipeline_pdf():
    sample_pdf = Path("./data/sample_docs/Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not present in environment")

    pipeline = IngestionPipeline(parent_size=1000, child_size=250)
    chunks_dict = pipeline.process_file(str(sample_pdf))

    assert "parents" in chunks_dict
    assert "children" in chunks_dict
    assert len(chunks_dict["children"]) > 0
    assert chunks_dict["children"][0]["metadata"]["source"] == sample_pdf.name
