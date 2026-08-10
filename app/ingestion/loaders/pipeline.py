from pathlib import Path
from typing import Any, Dict

from app.core.config import settings
from app.ingestion.chunking.chunker import ParentChildChunker
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.pdf_parser import PDFParser


class IngestionPipeline:
    """Orchestrates document parsing and Parent-Child chunking for medical documents."""

    def __init__(
        self,
        parent_size: int = settings.PARENT_CHUNK_SIZE,
        parent_overlap: int = settings.PARENT_CHUNK_OVERLAP,
        child_size: int = settings.CHILD_CHUNK_SIZE,
        child_overlap: int = settings.CHILD_CHUNK_OVERLAP,
    ):
        self.chunker = ParentChildChunker(
            parent_size=parent_size,
            parent_overlap=parent_overlap,
            child_size=child_size,
            child_overlap=child_overlap,
        )

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a PDF or DOCX file and generates linked Parent and Child chunks.
        Returns dict: {'parents': [...], 'children': [...]}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            parser = PDFParser(str(path))
        elif suffix in [".docx", ".doc"]:
            parser = DOCXParser(str(path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdf, .docx")

        # 1. Parse document pages/sections
        documents = parser.parse()

        # 2. Generate linked parent & child chunks
        chunks = self.chunker.create_chunks(documents)

        return chunks
