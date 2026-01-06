import re
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)


def extract_text_from_pdf(file):
    """
    Extract text from PDF file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        str: Extracted text
    """
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def extract_text_from_docx(file):
    """
    Extract text from DOCX file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        str: Extracted text
    """
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def extract_text_from_txt(file):
    """
    Extract text from TXT file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        str: Extracted text
    """
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        raise Exception(f"Error reading TXT: {str(e)}")


def extract_text(file, file_type):
    """
    Extract text from uploaded file based on file type.
    
    Args:
        file: Uploaded file object
        file_type (str): Type of file (pdf, docx, txt)
        
    Returns:
        str: Extracted text
    """
    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type == 'docx':
        return extract_text_from_docx(file)
    elif file_type == 'txt':
        return extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
