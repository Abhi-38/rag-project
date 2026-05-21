import fitz


class DocumentParser:

    @staticmethod
    def load_pdf(file_path: str):

        document = fitz.open(file_path)

        extracted_text = ""

        for page in document:

            extracted_text += page.get_text()

        return extracted_text