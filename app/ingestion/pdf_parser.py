import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
from pathlib import Path

class PDFParser:
    """Parses medical PDF documents with page number tracking and structural metadata."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def extract_pages(self) -> List[Dict[str, Any]]:
        """Extracts text page-by-page along with page numbers and detected headings."""
        doc = fitz.open(self.file_path)
        pages_data = []
        current_heading = "General"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Simple heuristic to detect chapter/section headings (all caps or short bold-like lines)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines[:3]:  # Check top 3 lines of page for headers
                if len(line) < 80 and (line.isupper() or re.match(r'^(CHAPTER|SECTION|PART|\d+\.)', line, re.I)):
                    current_heading = line
                    break

            if text.strip():
                pages_data.append({
                    "page": page_num + 1,
                    "source": self.file_path.name,
                    "heading": current_heading,
                    "text": text.strip()
                })

        doc.close()
        return pages_data
