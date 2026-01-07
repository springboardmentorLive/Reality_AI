import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing special characters, URLs, and numbers
    3. Tokenizing and removing stopwords
    4. Lemmatizing tokens
    """
    if not text:
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Key cleaning steps using regex
    text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
    text = re.sub(r'RT|cc', ' ', text)  # remove RT and cc
    text = re.sub(r'#\S+', '', text)  # remove hashtags
    text = re.sub(r'@\S+', ' ', text)  # remove mentions
    
    # Preserving special characters meaningful for technical skills: +, #, . (e.g. C++, C#, .NET, Node.js)
    # We remove punctuation that separates sentences but keep those inside words
    text = re.sub(r'[!"&\'(),:;<=>?@[\]^_`{|}~]', ' ', text) # remove standard punctuation except +, #, . - / % *
    
    text = re.sub(r'[^\x00-\x7f]', r' ', text) # remove non-ascii characters
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    
    # Tokenization (splitting into words)
    tokens = nltk.word_tokenize(text)
    
    # Stopword removal and Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1 # remove single chars
    ]
    
    return ' '.join(clean_tokens)
