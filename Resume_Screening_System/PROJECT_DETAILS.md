# Resume Screening System with ATS Capabilities

## 1. Project Overview
The **Resume Screening System** is a Machine Learning-powered application designed to automate the initial phase of the recruitment process. It analyzes resumes to predict the most suitable job profile, estimates an ATS (Applicant Tracking System) score, and provides visual insights into the candidate's fit for a role. This tool aims to assist HR professionals and recruiters in efficiently filtering through large volumes of applications.

## 2. Component Features

### Core Functionality
- **Multi-Format Support**: Accurately extracts text from PDF and TXT file formats.
- **Intelligent Preprocessing**: Cleans raw text by removing URLs, special characters, and stopwords to focus on meaningful content.
- **ML-Based Categorization**: Uses a **Logistic Regression** model trained on TF-IDF vectors to classify resumes into categories like *Data Scientist, Web Developer, Java Developer, Python Developer, HR, and DevOps*.

### Visual Analytics & Insights
- **Prediction Confidence**: Displays the model's confidence probability for the predicted role.
- **ATS Score Gauge**: Estimates a compatibility score (0-100) based on keyword density and content relevance.
- **Skill Match Analysis**:
    - *Keyword Match Chart*: A donut chart visualizing the percentage of matched keywords against detailed job descriptions.
    - *Probability Distribution*: A horizontal bar chart showing how strongly the resume aligns with other possible categories.

## 3. Advantages & Disadvantages

### Advantages
- **Efficiency & Speed**: Capable of processing and categorizing a resume in seconds, saving hours of manual review time.
- **Scalability**: Can handle any volume of resumes without fatigue or consistency errors.
- **Objectivity**: Reduces unconscious human bias by evaluating candidates purely based on the text data and trained metrics.
- **Data-Driven Insights**: Provides actionable visualizations (like the probability distribution) that give a deeper understanding of a candidate's versatility beyond a single label.
- **Instant Feedback**: Useful for candidates to self-assess their resumes against standard job descriptions.

### Disadvantages
- **Keyword Dependency**: The system relies heavily on specific keywords. A qualified candidate might be ranked lower if they use synonyms that the model doesn't recognize.
- **Formatting Sensitivity**: While robust, the PDF parser may struggle with highly complex, multi-column, or graphic-heavy resume layouts.
- **Context Limitations**: As a text-based ML model, it may not fully grasp complex project descriptions, leadership nuances, or "soft skills" that are often better evaluated by a human.
- **Domain Limit**: The current model is trained on specific IT and HR roles. It requires retraining to be effective in other domains (e.g., Medicine, Law).

## 4. Purpose
The primary purpose of this project is to **bridge the gap between massive applicant volumes and limited recruiter time**. By automating the "screening" phase:
1.  **Recruiters** receive a shortlist of the most relevant candidates.
2.  **Candidates** can receive faster feedback or optimize their resumes for standard ATS systems.
3.  **Organizations** improve their hiring efficiency and standardization.

## 5. Technical Architecture

### Technology Stack
- **Language**: Python 3.x
- **Frontend Framework**: Streamlit (for a responsive, interactive web UI).
- **Machine Learning**: Scikit-Learn (Logistic Regression, TF-IDF Vectorizer).
- **Data Processing**: Pandas, NumPy.
- **Natural Language Processing**: NLTK (for stopwords and lemmatization), Regex.
- **Visualization**: Plotly (for interactive gauges and charts).

### Project Structure
- `app.py`: The main application entry point controlling the UI and logic flow.
- `train.py`: Script to train the ML model and save artifacts (`resume_model.pkl`, `tfidf.pkl`).
- `utils/`:
    - `text_cleaner.py`: Handles regex-based text sanitation.
    - `resume_parser.py`: Wrappers for `pdfplumber` or similar libraries to extract text.
    - `skill_extractor.py`: Helper to identify specific technical skills from the text.

## 6. Future Scope
- **Deep Learning Integration**: Implementing BERT or Transformers for better context understanding.
- **Resume Ranking**: allowing bulk upload to rank multiple candidates against a single Job Description.
- **User Accounts**: Creating a history of scanned resumes for recruiters.
- **API Deployment**: Exposing the model as a REST API for integration with existing HR portals.
