import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


class PDFParser:
    """Extracts text page-by-page from PDF documents preserving page numbers, heading, and source metadata."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    def parse(self) -> List[Dict[str, Any]]:
        """Parses PDF pages and returns list of document page dicts."""
        doc = fitz.open(self.file_path)
        extracted_pages = []
        current_heading = "General"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines[:3]:
                if len(line) < 80 and (
                    line.isupper() or re.match(r"^(CHAPTER|SECTION|PART|\d+\.)", line, re.I)
                ):
                    current_heading = line
                    break

            if text:
                extracted_pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": self.file_path.name,
                            "page": page_num + 1,
                            "heading": current_heading,
                            "file_type": "pdf",
                        },
                    }
                )

        doc.close()
        return extracted_pages
