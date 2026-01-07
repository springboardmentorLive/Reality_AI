import streamlit as st
import pickle
import os
import pandas as pd
import time
import base64
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from utils.resume_parser import extract_text_from_pdf, extract_text_from_txt
from utils.text_cleaner import clean_text
from utils.skill_extractor import extract_skills, SKILL_DB
import traceback

# --- Page Config ---
st.set_page_config(
    page_title="Resume Screening & ATS System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #ffffff !important;
        }

        /* App Background handled by Python function */

        [data-testid="stHeader"] {
            background: transparent;
        }
        
        /* Navigation Bar Customization */
        .nav-link-selected {
            background-color: #6366f1 !important;
        }

        /* Headings */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        }

        h1 {
            background: linear-gradient(to right, #a5b4fc, #e879f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none; /* Gradient text doesn't look good with shadow, but we made it lighter */
        }

        p, li, span, div {
            color: #f1f5f9 !important;
        }

        /* Cards */
        .card,
        div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
            background: rgba(15, 23, 42, 0.95); /* High opacity for readability */
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 2rem;
        }

        /* Buttons */
        .stButton > button {
            height: 48px;
            border-radius: 12px;
            font-weight: 600;
            border: none;
            color: white !important;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            box-shadow: 0 4px 12px rgba(99,102,241,0.4);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99,102,241,0.6);
        }

        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
            color: white !important;
        }

        .stButton > button p {
            color: white !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 14px;
            padding: 20px;
            border: 2px dashed #475569;
            transition: border-color 0.3s;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #818cf8;
        }

        /* Metric Boxes */
        .metric-box {
            text-align: center;
            padding: 1.5rem;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #818cf8;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s ease;
        }

        .metric-box:hover {
            transform: translateY(-5px);
            background: rgba(30, 41, 59, 1);
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        
        .metric-label {
            color: #cbd5e1 !important;
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Skill Chips */
        .skill-chip {
            background: rgba(99, 102, 241, 0.25);
            color: #e0e7ff !important;
            border: 1px solid rgba(99, 102, 241, 0.5);
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0 6px 6px 0;
            display: inline-block;
        }

        .skill-chip.missing {
            background: rgba(244, 63, 94, 0.25);
            color: #ffe4e6 !important;
            border-color: rgba(244, 63, 94, 0.5);
        }

        .skill-chip.match {
            background: rgba(34, 197, 94, 0.25);
            color: #dcfce7 !important;
            border-color: rgba(34, 197, 94, 0.5);
        }
        
        /* Text Area & Inputs */
        .stTextArea textarea {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
            border-radius: 10px;
        }
        
        .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 1px #6366f1 !important;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #0f172a; 
        }

        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #475569; 
        }
    </style>
    """, unsafe_allow_html=True)



def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.95)), url("data:image/jpg;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

local_css()
set_background('resume_bg.jpg')

# --- Load Resources ---
@st.cache_resource
def load_models():
    if not os.path.exists('model/resume_model.pkl') or not os.path.exists('model/tfidf.pkl'):
        return None, None
    model = pickle.load(open('model/resume_model.pkl', 'rb'))
    tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_models()




# --- Helper Functions ---
def clean_job_description(jd):
    """Simple cleaning for job description to comparing keywords"""
    if not jd: return []
    clean = clean_text(jd)
    return set(clean.split())

def create_gauge_chart(score):
    """Creates a gauge chart for ATS score using Plotly"""
    # Balanced Palette for Dark Mode
    color = "#ef4444" # Red
    if score >= 80: color = "#2dd4bf" # Teal
    elif score >= 50: color = "#fbbf24" # Amber
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "ATS Score", 'font': {'size': 24, 'color': '#f8fafc', 'weight': 700}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(45, 212, 191, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#f8fafc", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Outfit"})
    return fig

def create_probability_chart(probs, classes):
    """Creates a horizontal bar chart for all class probabilities"""
    df = pd.DataFrame({'Role': classes, 'Probability': probs * 100})
    df = df.nlargest(10, 'Probability')
    df = df.sort_values('Probability', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Role'],
        orientation='h',
        marker=dict(color=df['Probability'], colorscale='Plasma', showscale=False) # Plasma fits dark mode well
    ))
    
    fig.update_layout(
        title={'text': "Role Prediction Confidence", 'font': {'size': 20, 'color': '#f8fafc', 'weight': 'bold'}},
        xaxis_title="Confidence (%)",
        height=500,
        margin=dict(l=0, r=10, t=50, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False, tickfont=dict(size=12, family="Outfit", color='#e2e8f0')),
        xaxis=dict(
            range=[0, 100], 
            showgrid=True,
            gridcolor='#334155',
            dtick=10,
            title_font=dict(color='#94a3b8'),
            tickfont=dict(color='#94a3b8')
        )
    )
    return fig

def create_keyword_chart(match_count, total_count):
    """Creates a donut chart for keyword matching"""
    missing = max(0, total_count - match_count)
    
    fig = go.Figure(data=[go.Pie(
        labels=['Matched', 'Missing'],
        values=[match_count, missing],
        hole=.6,
        marker_colors=['#2dd4bf', '#fb923c'], 
        textinfo='label+percent',
        textfont_size=16,
        textfont_color='white',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title={'text': "Keyword Match Ratio", 'font': {'size': 20, 'color': '#f8fafc', 'weight': 'bold'}},
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#e2e8f0'))
    )
    return fig

# --- Views ---

def home_page():
    # --- Header ---
    st.markdown("""
    <div>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>Resume Screening System</h1>
        <p style='color: #cbd5e1; margin-bottom: 2rem; font-size: 1.1rem;'>Using advanced NLP to match resumes with job descriptions.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ System Error: Models not found. Please run `python train.py` first.")
        st.stop()

    # --- Upload Section ---
    col_upload, col_jd = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.markdown("### 📤 Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=['pdf', 'txt'])
        
    with col_jd:
        st.markdown("### 📝 Job Description")
        job_description = st.text_area("Paste the job description here", height=150, placeholder="e.g. We are looking for a Data Scientist with Python, SQL, and Machine Learning experience...")


    # --- Analyze Action ---
    if uploaded_file:
        col_act1, col_act2, col_act3 = st.columns([1, 2, 1])
        with col_act2:
            analyze_btn = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)
            
        if analyze_btn:
            try:
                with st.spinner("🔍 Parsing resume and analyzing patterns..."):
                    time.sleep(1.5) # UX Delay
                    
                    # --- PROCESSING ---
                    # 1. Extract Text
                    if uploaded_file.type == "application/pdf":
                        raw_text = extract_text_from_pdf(uploaded_file)
                    else:
                        raw_text = extract_text_from_txt(uploaded_file)
                    
                    if not raw_text:
                        st.error("Text extraction failed.")
                        st.stop()
                        
                    clean_resume = clean_text(raw_text)
                    
                    # 2. Extract Skills
                    resume_skills = extract_skills(clean_resume)
                    
                    # 3. Predict Role
                    features = tfidf.transform([clean_resume])
                    probs = model.predict_proba(features)[0]
                    top_idx = probs.argmax()
                    predicted_role = model.classes_[top_idx]
                    predicted_role_title = predicted_role
                    confidence = probs[top_idx] * 100
                    
                    # Confidence Threshold Check
                    if confidence < 20:
                        st.warning(f"⚠️ Low confidence ({confidence:.1f}%). The system is unsure about this resume.")
                        # predicted_role_title = "Uncertain"
                    else:
                        st.success(f"✅ Prediction confident: {confidence:.1f}%")
                    
                    # 4. Job Match (If JD provided)
                    jd_keywords = set()
                    if job_description:
                        jd_keywords = clean_job_description(job_description) # Very basic tokenization
                        
                        # Calculate Jaccard Similarity (Simple ATS Score proxy)
                        resume_tokens = set(clean_text(raw_text).split())
                        
                        intersection = resume_tokens.intersection(jd_keywords)
                        if len(jd_keywords) > 0:
                            ats_score_raw = (len(intersection) / len(jd_keywords)) * 100 * 2.5 # Multiply for leniency
                            ats_score = min(int(ats_score_raw), 100)
                        else:
                            ats_score = int(confidence) # Fallback to confidence
                            
                        missing_keywords = list(jd_keywords - resume_tokens)
                    else:
                        ats_score = int(confidence) # Use model confidence as a proxy score if no JD
                        missing_keywords = []

                    # Special handling for 'Other' role
                    if predicted_role == 'Other':
                        st.warning("⚠️  Role Not Identified: The resume appears to be for a role not currently supported by this system (e.g., Medical, Teaching, General).")
                        ats_score = 0 # Low score for unsupported roles

                    # --- DASHBOARD ---
                    st.markdown("---")
                    
                    # ATS Score Row
                    st.subheader("🎯 Analysis Results")
                    
                    score_col, metrics_col = st.columns([1, 2])
                    
                    with score_col:
                        st.plotly_chart(create_gauge_chart(ats_score), use_container_width=True)
                        
                    with metrics_col:
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{predicted_role_title}</div>
                                <div class="metric-label">Predicted Role</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{len(resume_skills)}</div>
                                <div class="metric-label">Skills Detected</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m3:
                            experience_placeholder = "Fresher" # Future: Extract exp
                            st.markdown(f"""
                            <div class="metric-box">
                                <div class="metric-value">{experience_placeholder}</div>
                                <div class="metric-label">Experience Level</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # --- Visualizations ---
                    st.markdown("### 📊 Visual Insights")
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.plotly_chart(create_probability_chart(probs, model.classes_), use_container_width=True)
                        
                    with viz_col2:
                        if job_description:
                            # Ensure variables are available from previous scope
                            match_count = len(intersection)
                            total_count = len(jd_keywords)
                            st.plotly_chart(create_keyword_chart(match_count, total_count), use_container_width=True)
                        else:
                            st.info("ℹ️  Paste a Job Description to see keyword matching analysis.")

                    # Split View: Preview vs Suggestions
                    st.markdown("### 📄 Detailed Reviews")
                    col_preview, col_suggestions = st.columns([1, 1], gap="medium")
                    
                    with col_preview:
                        st.markdown("#### Resume Preview")
                        
                        display_text = raw_text[:3000] + "..." if len(raw_text) > 3000 else raw_text
                        
                        annotated_html = ""
                        for line in display_text.split('\n'):
                            if not line.strip(): continue
                            line_html = line
                            
                            # Highlight found skills
                            for skill in resume_skills:
                                if skill.lower() in line.lower():
                                    line_html = f'<span style="background-color: rgba(34, 197, 94, 0.2); color: #86efac; border-radius: 4px; padding: 0 4px;">{line}</span>'
                                    break
                            
                            annotated_html += f"<div>{line_html}</div><br>"

                        st.markdown(
                            f"""
                            <div style="height: 500px; overflow-y: scroll; padding: 20px; background: #1e293b; border: 1px solid #334155; border-radius: 12px; font-size: 0.9rem; font-family: 'Courier New', monospace; color: #cbd5e1;">
                                {annotated_html}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )


                    with col_suggestions:
                        st.markdown("#### 💡 Suggestions")
                        
                        # 1. Missing Keywords (from JD)
                        if job_description:
                            st.write("**Missing Keywords (Critical):**")
                            if missing_keywords:
                                # Show top 15 missing
                                chips = ""
                                for word in missing_keywords[:15]:
                                    if len(word) > 3: # Filter tiny words
                                        chips += f'<span class="skill-chip missing">{word}</span>'
                                st.markdown(chips, unsafe_allow_html=True)
                            else:
                                st.success("🎉 Great job! You matched most keywords.")
                            st.markdown("---")

                        # 2. Skill Gaps (General for role)
                        st.write(f"**Recommended Skills for {predicted_role_title}:**")
                        
                        if predicted_role == 'Other':
                             st.info("ℹ️  Since the role is 'Other / Non-Technical', no specific technical skill gaps could be determined. Please manually review the job description.")
                        else:
                            all_cat_skills = SKILL_DB.get(predicted_role_title, [])
                            missing_skills = [s for s in all_cat_skills if s not in resume_skills]
                            
                            if missing_skills:
                                chips = ""
                                for s in missing_skills:
                                    chips += f'<span class="skill-chip missing">{s}</span>'
                                st.markdown(chips, unsafe_allow_html=True)
                            else:
                                st.info("No specific skill gaps identified for this role.")

                        st.markdown("---")
                        
                        # 3. Formatting Check (Static/Mock for now)
                        st.write("**Formatting Score:**")
                        st.info("ℹ️  PDF content parsed successfully. Avoid using tables or columns for better ATS parsing.")
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                with st.expander("Show specific error details"):
                    st.code(traceback.format_exc())


def skill_page():
    st.markdown("## 🧠 Skill Database")
    st.markdown("This database is used to identify known technical skills in the resume.")
    
    st.markdown("---")
    
    for category, skills in SKILL_DB.items():
        with st.expander(f"📌 {category}", expanded=True):
            chips = ""
            for s in skills:
                chips += f'<span class="skill-chip match">{s}</span>'
            st.markdown(chips, unsafe_allow_html=True)

def about_page():
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    This **Resume Screening AI** is designed to help job seekers and recruiters by automatically analyzing resumes against job descriptions.
    
    ### How It Works
    1.  **Text Extraction**: We use `PyPDF2` to read your resume.
    2.  **Cleaning**: Advanced text cleaning removes noise (special characters, stop words).
    3.  **Skill Extraction**: A keyword matching system identifies technical skills.
    4.  **Role Prediction**: A trained Machine Learning model predicts the job role.
    5.  **Data Visualization**: Plotly charts visualize the match confidence and keyword overlap.
    
    ### Technologies Used
    -   **Python 3.9+**
    -   **Streamlit** (Frontend)
    -   **Scikit-Learn** (ML Model)
    -   **Plotly** (Charts)
    -   **NLTK** (Text Processing)
    
    ---
    *Built with ❤️ for the Springboard Project.*
    """)

# --- Main Layout ---
def main():
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Skill DB", "About"],
        icons=["house", "cpu", "info-circle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(15, 23, 42, 0.9)"},
            "icon": {"color": "#818cf8", "font-size": "16px"}, 
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#312e81"},
            "nav-link-selected": {"background-color": "#6366f1"},
        }
    )
    
    if selected == "Dashboard":
        home_page()
    elif selected == "Skill DB":
        skill_page()
    elif selected == "About":
        about_page()

if __name__ == "__main__":
    main()
