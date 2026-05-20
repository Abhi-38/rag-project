from app.ingestion.parser import DocumentParser


text = DocumentParser.load_pdf(
    "data/sample_docs/Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf"
)

print(text[:3000])