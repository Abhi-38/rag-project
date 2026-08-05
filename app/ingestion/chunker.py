import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

class ParentChildChunker:
    """Implements Parent-Child chunking for medical documents."""
    
    def __init__(self, 
                 parent_size: int = settings.PARENT_CHUNK_SIZE,
                 parent_overlap: int = settings.PARENT_CHUNK_OVERLAP,
                 child_size: int = settings.CHILD_CHUNK_SIZE,
                 child_overlap: int = settings.CHILD_CHUNK_OVERLAP):
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Splits page data into linked Parent and Child chunks.
        Returns dict containing 'parents' dict (keyed by parent_id) and 'children' list.
        """
        parents_map = {}
        children_list = []

        for page_info in pages_data:
            text = page_info["text"]
            page_num = page_info["page"]
            source = page_info["source"]
            heading = page_info["heading"]

            # Split into parent chunks
            parent_texts = self.parent_splitter.split_text(text)
            
            for p_idx, p_text in enumerate(parent_texts):
                parent_id = f"{source}_p{page_num}_{p_idx}_{uuid.uuid4().hex[:6]}"
                
                parent_obj = {
                    "id": parent_id,
                    "text": p_text,
                    "metadata": {
                        "source": source,
                        "page": page_num,
                        "heading": heading,
                        "type": "parent"
                    }
                }
                parents_map[parent_id] = parent_obj

                # Split parent text into child chunks
                child_texts = self.child_splitter.split_text(p_text)
                for c_idx, c_text in enumerate(child_texts):
                    child_id = f"{parent_id}_c{c_idx}"
                    child_obj = {
                        "id": child_id,
                        "parent_id": parent_id,
                        "text": f"[Context: {heading} (Page {page_num})]\n{c_text}",
                        "raw_text": c_text,
                        "metadata": {
                            "parent_id": parent_id,
                            "source": source,
                            "page": page_num,
                            "heading": heading,
                            "type": "child"
                        }
                    }
                    children_list.append(child_obj)

        return {
            "parents": parents_map,
            "children": children_list
        }
