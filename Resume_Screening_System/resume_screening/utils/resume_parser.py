import PyPDF2

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file uploaded via Streamlit.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_txt(uploaded_file):
    """
    Extracts text from a TXT file uploaded via Streamlit.
    """
    try:
        # Streamlit UploadedFile is bytes-like, need to decode
        return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        return f"Error reading Text file: {str(e)}"
