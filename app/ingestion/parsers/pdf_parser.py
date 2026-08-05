import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path

class PDFParser:
    """Extracts text page-by-page from PDF documents preserving page numbers and source metadata."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def parse(self) -> List[Dict[str, Any]]:
        """Parses PDF pages and returns list of document page dicts."""
        doc = fitz.open(self.file_path)
        extracted_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            
            if text:
                extracted_pages.append({
                    "text": text,
                    "metadata": {
                        "source": self.file_path.name,
                        "page": page_num + 1,
                        "file_type": "pdf"
                    }
                })

        doc.close()
        return extracted_pages
