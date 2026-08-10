import re
from pathlib import Path
from typing import Any, Dict, List


class DOCXParser:
    """Extracts text from Microsoft Word (.docx) documents."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")

    def parse(self) -> List[Dict[str, Any]]:
        """Parses DOCX document paragraphs into document dicts."""
        heading = "General"
        try:
            import docx

            doc = docx.Document(self.file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                for p in paragraphs[:3]:
                    if len(p) < 80 and (
                        p.isupper() or re.match(r"^(CHAPTER|SECTION|PART|\d+\.)", p, re.I)
                    ):
                        heading = p
                        break
            full_text = "\n\n".join(paragraphs)
        except ImportError:
            import xml.etree.ElementTree as ET
            import zipfile

            with zipfile.ZipFile(self.file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            paragraphs = []
            for elem in tree.iter():
                if elem.tag.endswith("}t") and elem.text:
                    paragraphs.append(elem.text)
            full_text = " ".join(paragraphs)

        if not full_text.strip():
            return []

        return [
            {
                "text": full_text.strip(),
                "metadata": {
                    "source": self.file_path.name,
                    "page": None,
                    "heading": heading,
                    "file_type": "docx",
                },
            }
        ]
