import streamlit as st
import pandas as pd
import pickle
import base64
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)

# --------------------------------------------------
# LOAD BACKGROUND IMAGE
# --------------------------------------------------
def get_base64_bg(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_bg("assets/bg1.jpg")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
        url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
    font-family: 'Arial', sans-serif;
}}

.block-container {{
    padding: 2rem 5%;
    margin: 0;
    background: transparent;
    width: 100%;
}}

h1, h2, h3, h4, h5, h6 {{
    text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
}}

p, li {{
    color: #f0f0f0;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.6);
}}

.stButton>button {{
    background-color: #1f2937;
    color: white;
    border: none;
    padding: 0.5rem 1.2rem;
    border-radius: 5px;
    font-weight: bold;
}}

.stButton>button:hover {{
    background-color: #374151;
    cursor: pointer;
}}

.result-approve {{
    color: #20c997;  /* teal green */
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(32,201,151,0.3);
    box-shadow: 0 4px 15px rgba(32,201,151,0.4);
}}

.result-reject {{
    color: #ff6b6b;  /* coral red */
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(255,107,107,0.3);
    box-shadow: 0 4px 15px rgba(255,107,107,0.4);
}}

/* Footer */
footer {{
    position: fixed;
    bottom: 0;
    width: 100%;
    background: rgba(0,0,0,0.6);
    color: #f0f0f0;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    z-index: 999;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.4);
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL & ENCODERS
# --------------------------------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
le_emp = pickle.load(open("employment_encoder.pkl", "rb"))
le_area = pickle.load(open("area_encoder.pkl", "rb"))

# --------------------------------------------------
# NAV STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# --------------------------------------------------
# NAVBAR
# --------------------------------------------------
st.markdown("""
<div style="position:fixed;top:0;width:100%;height:60px;background:rgba(0,0,0,0.8);display:flex;align-items:center;padding:0 40px;z-index:999;">
    <div style="color:white;font-size:22px;font-weight:600;">💼 Loan Approval Prediction</div>
</div>
<br><br>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("Prediction"):
        st.session_state.page = "Predict"
with col3:
    if st.button("Analytics"):
        st.session_state.page = "Analytics"
with col4:
    if st.button("About"):
        st.session_state.page = "About"

st.markdown("<br><br>", unsafe_allow_html=True)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if st.session_state.page == "Home":
    st.markdown("<h1>Loan Approval Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3>AI-powered loan eligibility assessment system</h3>", unsafe_allow_html=True)

    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=260)

    st.markdown("""
### Project Overview
This application uses **Machine Learning** to predict whether a loan application is likely to be **approved or rejected**
based on applicant details.

### Input Parameters
- Applicant Income  
- Credit Score  
- Loan Amount  
- Employment Type  
- Number of Dependents  
- Loan Term  
- Property Area  

### Purpose
This project demonstrates how **Machine Learning models** can assist in financial decision-making and risk assessment.

**Disclaimer:**  
This application is created only for **educational and demonstration purposes**.
Predictions should not be used for real financial decisions.
""")

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif st.session_state.page == "Predict":
    st.markdown("<h1>Loan Eligibility Check</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Income", min_value=0)
        credit = st.number_input("Credit Score", 300, 900, 650)
        loan = st.number_input("Loan Amount", min_value=0)
        emp = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
    with col2:
        dep = st.number_input("Dependents", 0, 10, 0)
        term = st.slider("Loan Term (Months)", 12, 360, 120)
        area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict"):
        emp_val = le_emp.transform([emp])[0]
        area_val = le_area.transform([area])[0]

        df = pd.DataFrame([[income, credit, loan, emp_val, dep, term, area_val]],
            columns=[
                "Income", "CreditScore", "LoanAmount",
                "EmploymentType", "Dependents",
                "LoanTerm", "PropertyArea"
            ])

        pred = model.predict(df)[0]
        prob = model.predict_proba(df).max() * 100

        # Save prediction for analytics
        st.session_state.predictions.append(pred)

        if pred == 1:
            st.markdown(f"<div class='result-approve'>Loan Approved ({prob:.2f}%)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-reject'>Loan Rejected ({prob:.2f}%)</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ANALYTICS PAGE
# --------------------------------------------------
elif st.session_state.page == "Analytics":
    st.markdown("<h1>Loan Analytics Dashboard</h1>", unsafe_allow_html=True)

    approvals = st.session_state.predictions.count(1)
    rejections = st.session_state.predictions.count(0)
    total = approvals + rejections

    if total == 0:
        st.info("No predictions yet. Please check some applicants on the Prediction page.")
    else:
        # Summary cards
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""
            <div style='
                background-color: rgba(32,201,151,0.3);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            '>
            <h2 style='color:#20c997'>{approvals}</h2>
            <p>Approved Loans</p>
            </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
            <div style='
                background-color: rgba(255,107,107,0.3);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            '>
            <h2 style='color:#ff6b6b'>{rejections}</h2>
            <p>Rejected Loans</p>
            </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
            <div style='
                background-color: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            '>
            <h2 style='color:white'>{total}</h2>
            <p>Total Applications</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align:center;color:white'>Approval Distribution</h3>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots()
            ax1.pie([approvals, rejections],
                    labels=["Approved", "Rejected"],
                    colors=["#20c997", "#ff6b6b"],
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"edgecolor": "black"})
            ax1.axis("equal")
            st.pyplot(fig1, clear_figure=True)

        with col2:
            st.markdown("<h3 style='text-align:center;color:white'>Approval Count</h3>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots()
            ax2.bar(["Approved", "Rejected"], [approvals, rejections], color=["#20c997", "#ff6b6b"])
            ax2.set_ylabel("Number of Applications", color="white", fontsize=12)
            ax2.set_facecolor("none")
            ax2.tick_params(axis='x', colors='white', labelsize=12)
            ax2.tick_params(axis='y', colors='white', labelsize=12)
            for spine in ax2.spines.values():
                spine.set_color('white')
            st.pyplot(fig2, clear_figure=True)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif st.session_state.page == "About":
    st.markdown("<h1>About This Project</h1>", unsafe_allow_html=True)

    st.markdown("""
### Loan Approval Prediction System
The Loan Approval Prediction App is a **Machine Learning–based system**
that evaluates loan eligibility using historical data and predictive modeling.

### Machine Learning Model
- Algorithm Used: Logistic Regression  
- Problem Type: Binary Classification  
- Output:
  - Approved (1)
  - Rejected (0)

### Technology Stack
- Frontend: Streamlit  
- Programming Language: Python  
- ML Library: Scikit-learn  
- Data Handling: Pandas  

### Learning Outcomes
- End-to-end Machine Learning workflow  
- Feature preprocessing and encoding  
- Model training and deployment  
- Building interactive ML web applications  

This project is suitable for **students, portfolios, and placement demonstrations**.
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<footer>
Loan Approval Prediction App | Prasanna Kuchipudi | © 2026
</footer>
""", unsafe_allow_html=True)
