import uuid
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings


class ParentChildChunker:
    """
    Implements canonical Parent-Child document chunking.
    Parent chunks provide broad context for generation; Child chunks provide precision for search.
    """

    def __init__(
        self,
        parent_size: int = settings.PARENT_CHUNK_SIZE,
        parent_overlap: int = settings.PARENT_CHUNK_OVERLAP,
        child_size: int = settings.CHILD_CHUNK_SIZE,
        child_overlap: int = settings.CHILD_CHUNK_OVERLAP,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

    def create_chunks(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Splits input parsed documents into linked Parent and Child chunk lists.
        Input documents: List of dicts with 'text' and 'metadata' ({source, page, heading, file_type}).
        Output dict: {'parents': [...], 'children': [...]}
        """
        parents_list = []
        children_list = []

        for doc in documents:
            text = doc.get("text", "")
            base_meta = doc.get("metadata", {}).copy()

            if not text.strip():
                continue

            source = base_meta.get("source", "doc")
            page = base_meta.get("page")
            page_str = f"p{page}" if page is not None else "pNA"

            # 1. Split document into parent chunks
            parent_texts = self.parent_splitter.split_text(text)

            for p_idx, p_text in enumerate(parent_texts):
                parent_id = f"{source}_{page_str}_p{p_idx}_{uuid.uuid4().hex[:6]}"

                parent_meta = base_meta.copy()
                parent_meta["parent_id"] = parent_id
                parent_meta["type"] = "parent"
                parent_meta = {k: v for k, v in parent_meta.items() if v is not None}

                parent_obj = {
                    "parent_id": parent_id,
                    "text": p_text,
                    "metadata": parent_meta,
                }
                parents_list.append(parent_obj)

                # 2. Split parent chunk into child chunks
                child_texts = self.child_splitter.split_text(p_text)
                for c_idx, c_text in enumerate(child_texts):
                    child_id = f"{parent_id}_c{c_idx}"

                    child_meta = base_meta.copy()
                    child_meta["child_id"] = child_id
                    child_meta["parent_id"] = parent_id
                    child_meta["type"] = "child"
                    child_meta = {k: v for k, v in child_meta.items() if v is not None}

                    heading_str = base_meta.get("heading", "General")
                    page_label = f"Page {page}" if page is not None else "Doc"
                    formatted_text = f"[Context: {heading_str} ({page_label})]\n{c_text}"

                    child_obj = {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "text": formatted_text,
                        "raw_text": c_text,
                        "metadata": child_meta,
                    }
                    children_list.append(child_obj)

        return {"parents": parents_list, "children": children_list}
