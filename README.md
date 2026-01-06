# Resume Screening System

## 📋 Project Overview
This project is a Machine Learning-powered web application designed to automate the initial screening process of recruitment. It accepts resumes in various formats (PDF, DOCX, TXT) and predicts the most suitable job role for the candidate based on the resume content.

The system currently supports the identification of three specific roles:
*   **Data Scientist**
*   **Java Developer**
*   **Web Developer**

## 🏗️ Architecture & Implementation

The project follows a standard Machine Learning pipeline integrated into a Streamlit web application.

### 1. Data Preparation & Model Training (`train_model.py`)
*   **Dataset:** A synthetic dataset was created with representative keywords and phrases for each of the three job roles.
*   **Preprocessing:** The raw text data undergoes cleaning to remove noise (URLs, emails, special characters) and is normalized (lowercasing).
*   **Feature Extraction:** We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text data into numerical vectors. This highlights unique, role-specific keywords.
*   **Modeling:** A **Random Forest Classifier** is trained on the vectorized data. This ensemble method provides robust predictions and handles high-dimensional text data well.
*   **Artifacts:** The trained model and vectorizer are serialized and saved as `.pkl` files for use in the app.

### 2. Utility Layer (`utils.py`)
This module handles all text processing tasks:
*   **Text Extraction:** Specialized functions to parse different file formats:
    *   `PyPDF2` for PDF files
    *   Standard file reading for Text files.
*   **Text Cleaning:** A robust `clean_text` function that:
    *   Removes URLs, emails, and special characters.
    *   Tokenizes text.
    *   Removes English stop words (common words like "the", "is", "at").
    *   Performs **Lemmatization** (converting words to their base form, e.g., "coding" to "code") using NLTK.

### 3. Application Interface (`app.py`)
The user interface is built with **Streamlit**:
*   **File Upload:** Accepts files and validates the extension.
*   **Real-time Analysis:**
    1.  The uploaded file is passed to the extraction logic.
    2.  The text is cleaned using the same logic as the training phase.
    3.  The text is vectorized using the loaded TF-IDF object.
    4.  The Pre-trained Random Forest model predicts the class probabilities.
*   **Visualization:**
    *   Displays the predicted role prominently.
    *   Shows confidence scores for all potential roles.
    *   Provides text statistics (Character count, Word count) to verify total analysis.
    *   Allows users to inspect the full extracted text.

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8+
*   pip (Python package manager)

### Step 1: Clone/Download the Repository
Ensure you have the project files in a directory.

### Step 2: Install Dependencies
Install the required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: This will install Streamlit, scikit-learn, pandas, nltk, PyPDF2, and python-docx.*

### Step 3: Train the Model
Before running the app, you need to generate the model files. Run the training script:
```bash
python train_model.py
```
*   This will create `resume_classifier.pkl` and `tfidf_vectorizer.pkl`.
*   You will see an accuracy report in the terminal.

### Step 4: Run the Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

```
resume_parser/
├── app.py                      # Main Streamlit application
├── train_model.py              # Script to train and save the ML model
├── utils.py                    # Helper functions for text processing
├── requirements.txt            # List of Python dependencies
├── resume_classifier.pkl       # Saved Random Forest model (generated)
├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer (generated)
├── README.md                   # Project documentation
└── [sample_resumes]            # Text files for testing
```

## 🧪 Usage Instructions

1.  **Open the App:** Navigate to the URL provided by Streamlit.
2.  **Upload a Resume:** Use the file uploader widget to select a PDF, DOCX, or TXT resume.
3.  **View Results:**
    *   **Predicted Role:** The system's best guess.
    *   **Confidence Scores:** Probability distribution across all supported roles.
    *   **Stats:** Verify that the full text length was processed.
4.  **Inspect Text:** Expand the "View Full Extracted Text" section to see exactly what the model "saw".
