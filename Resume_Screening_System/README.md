# Resume Screening System

A machine learning-based resume screening app built with Python and Streamlit.
This system extracts text from resumes (PDF/TXT), analyzes the content using TF-IDF and Logistic Regression, and predicts the most suitable job profile.

> **[View Detailed Project Documentation](PROJECT_DETAILS.md)** for in-depth information on Advantages, Disadvantages, Purpose, and Architecture.

## Dataset
To train with real data, use the **"Updated Resume Dataset"** from Kaggle.
- **Required Columns**: `Resume`, `Category`.
- **File Name**: Save the download as `UpdatedResumeDataSet.csv` in the `resume_screening/` folder.
- **NOTE**: The system currently includes a **synthetic dataset generator** so you can run and test the app immediately without downloading files.

## Features
- **Upload**: PDF or TXT resumes.
- **Preprocessing**: Cleans text (removes URLs, punctuation, stopwords).
- **Model**: Logistic Regression with TF-IDF vectorization.
- **Output**: Predicted Category + Confidence Score.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (generates artifacts in `model/`):
   ```bash
   python train.py
   ```
3. Run the App:
   ```bash
   streamlit run app.py
   ```
