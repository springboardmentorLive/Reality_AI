import streamlit as st
import pickle
import os
from utils import extract_text, clean_text

# Page configuration
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .prediction-title {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .prediction-result {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    try:
        with open('resume_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

def predict_job_role(text, model, vectorizer):
    """
    Predict job role from resume text.
    
    Args:
        text (str): Resume text
        model: Trained classifier
        vectorizer: TF-IDF vectorizer
        
    Returns:
        str: Predicted job role
    """
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    return prediction, probabilities

def main():
    # Header
    st.title("📄 Resume Screening System")
    st.markdown("### Upload a resume and get instant job role predictions")
    
    # Check if model exists
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("⚠️ Model files not found! Please run `train_model.py` first to train the model.")
        st.info("Run the following command in your terminal:\n```bash\npython train_model.py\n```")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system uses **Machine Learning** and **NLP** to analyze resumes and predict suitable job roles.
        
        **Supported Roles:**
        - 🔬 Data Scientist
        - ☕ Java Developer
        - 🌐 Web Developer
        
        **Supported Formats:**
        - PDF (.pdf)
        - Word (.docx)
        - Text (.txt)
        """)
        
        st.header("📊 How it works")
        st.markdown("""
        1. Upload your resume
        2. Text is extracted and cleaned
        3. ML model analyzes the content
        4. Get instant prediction
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload a resume in PDF, DOCX, or TXT format"
        )
    
    if uploaded_file is not None:
        # Display file info
        with col2:
            st.markdown("### File Info")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.markdown(f"**Type:** {uploaded_file.type}")
        
        # Extract file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            with st.spinner("🔍 Analyzing resume..."):
                # Extract text
                text = extract_text(uploaded_file, file_extension)
                
                if not text or len(text.strip()) < 50:
                    st.warning("⚠️ The extracted text is too short. Please upload a more detailed resume.")
                    return
                
                # Predict job role
                prediction, probabilities = predict_job_role(text, model, vectorizer)
                
                # Get class labels
                classes = model.classes_
                
            # Display prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">Predicted Job Role</div>
                    <div class="prediction-result">🎯 {prediction}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display confidence scores
            st.markdown("### 📊 Confidence Scores")
            
            # Create columns for each role
            cols = st.columns(3)
            
            # Role icons
            role_icons = {
                'Data Scientist': '🔬',
                'Java Developer': '☕',
                'Web Developer': '🌐'
            }
            
            for idx, (role, prob) in enumerate(zip(classes, probabilities)):
                with cols[idx]:
                    icon = role_icons.get(role, '📌')
                    st.metric(
                        label=f"{icon} {role}",
                        value=f"{prob * 100:.1f}%"
                    )
                    st.progress(prob)
            
            # Display text statistics
            st.markdown("### 📈 Resume Analysis Statistics")
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("Total Characters", f"{len(text):,}")
            with stat_cols[1]:
                word_count = len(text.split())
                st.metric("Total Words", f"{word_count:,}")
            with stat_cols[2]:
                cleaned_text = clean_text(text)
                cleaned_word_count = len(cleaned_text.split())
                st.metric("Processed Words", f"{cleaned_word_count:,}")
            
            st.info("✅ **Note:** The entire resume text was analyzed for prediction, not just a preview.")
            
            # Display extracted text preview
            with st.expander("📝 View Full Extracted Text"):
                st.text_area(
                    f"Complete Resume Text ({len(text)} characters)",
                    text,
                    height=400,
                    disabled=True
                )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure the file is not corrupted and contains readable text.")

if __name__ == "__main__":
    main()
