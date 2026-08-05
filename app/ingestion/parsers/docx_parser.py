from pathlib import Path
from typing import List, Dict, Any

class DOCXParser:
    """Extracts text from Microsoft Word (.docx) documents."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

    def parse(self) -> List[Dict[str, Any]]:
        """Parses DOCX document paragraphs into document dicts."""
        try:
            import docx
            doc = docx.Document(self.file_path)
            full_text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            # Fallback XML parsing if python-docx isn't installed
            import zipfile
            import xml.etree.ElementTree as ET
            
            with zipfile.ZipFile(self.file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for elem in tree.iter():
                if elem.tag.endswith('}t') and elem.text:
                    paragraphs.append(elem.text)
            full_text = " ".join(paragraphs)

        if not full_text.strip():
            return []

        return [{
            "text": full_text.strip(),
            "metadata": {
                "source": self.file_path.name,
                "page": 1,  # DOCX files do not have explicit PDF-style pagination
                "file_type": "docx"
            }
        }]
