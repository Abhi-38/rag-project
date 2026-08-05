import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    """
    Semantic Document Chunker.
    Uses sentence & paragraph boundaries with chunk_size=800 and chunk_overlap=150.
    Preserves document metadata (source, page, chunk_id).
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits input documents into semantically coherent chunks with preserved metadata.
        Input format: [{'text': '...', 'metadata': {'source': '...', 'page': 1}}]
        Output format: [{'chunk_id': '...', 'text': '...', 'metadata': {'source': '...', 'page': 1, 'chunk_id': '...'}}]
        """
        processed_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            base_metadata = doc.get("metadata", {}).copy()
            
            if not text.strip():
                continue

            split_texts = self.splitter.split_text(text)

            for idx, chunk_text in enumerate(split_texts):
                chunk_id = f"{base_metadata.get('source', 'doc')}_p{base_metadata.get('page', 1)}_c{idx}_{uuid.uuid4().hex[:6]}"
                
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_id"] = chunk_id

                processed_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_meta
                })

        return processed_chunks
