import sys
import time
from pathlib import Path
from app.api.routes.ingest import process_document_ingestion

def main():
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
    else:
        doc_path = "./data/sample_docs/Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf"

    path = Path(doc_path)
    if not path.exists():
        print(f"[ERROR] File does not exist at {doc_path}")
        sys.exit(1)

    start_time = time.time()
    print("=" * 60)
    print("Starting Canonical Medical RAG Ingestion Pipeline")
    print(f"Document: {path.name}")
    print("=" * 60)

    res = process_document_ingestion(str(path))
    elapsed = time.time() - start_time

    if res.get("status") == "error":
        print(f"\n[ERROR] Ingestion failed: {res.get('message')}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INGESTION SUCCESSFUL")
    print(f"- Document File: {res.get('file')}")
    print(f"- Parent Chunks Indexed: {res.get('parent_chunks_indexed')}")
    print(f"- Child Chunks Indexed: {res.get('child_chunks_indexed')}")
    print(f"- Total Vector DB Child Count: {res.get('vector_db_child_count')}")
    print(f"- Elapsed Time: {elapsed:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()
