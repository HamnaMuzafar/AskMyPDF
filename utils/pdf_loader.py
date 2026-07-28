import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    """
    Extract all text from an uploaded PDF file.
    """

    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    text = ""

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text