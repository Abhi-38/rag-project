import sys
from pathlib import Path
from app.api.routes.ingest import process_pdf_ingestion

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "./data/sample_docs/Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf"

    path = Path(pdf_path)
    if not path.exists():
        print(f"Error: PDF file does not exist at {pdf_path}")
        sys.exit(1)

    print(f"Starting Production RAG ingestion for: {path.name}")
    res = process_pdf_ingestion(str(path))
    print("\n--- Ingestion Result ---")
    for k, v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
