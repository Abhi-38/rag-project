from pathlib import Path
from typing import List, Dict, Any

from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.chunking.chunker import SemanticChunker

class IngestionPipeline:
    """Orchestrates parsing and semantic chunking of medical documents."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF or DOCX document and splits it into semantic chunks with preserved metadata.
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

        # 1. Parse document into pages/sections
        documents = parser.parse()

        # 2. Split into semantic chunks
        chunks = self.chunker.chunk_documents(documents)

        return chunks
