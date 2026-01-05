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

html, body, [class*="css"] {{
    background-color: transparent !important;
}}

.stApp {{
    background-image:
        linear-gradient(rgba(2,6,23,0.4), rgba(2,6,23,0.4)),
        url("data:image/png;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    padding: 2.5rem 6%;
}}

h1, h2, h3 {{
    color: #f8fafc !important;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.6);
}}

p, span, li {{
    color: #e5e7eb !important;
    font-size: 16px;
    line-height: 1.6;
}}

label {{
    color: #e5e7eb !important;
    font-weight: 600;
}}

input[type="number"], input[type="text"] {{
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    padding: 10px !important;
}}

div[data-baseweb="select"] > div {{
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}}

div[data-baseweb="select"] span {{
    color: #f8fafc !important;
}}

div[data-baseweb="slider"] {{
    color: #f8fafc !important;
}}

::placeholder {{
    color: #cbd5f5 !important;
}}

.stButton > button {{
    background-color: #111827;
    color: #f8fafc;
    font-weight: 600;
    padding: 10px 28px;
    border-radius: 10px;
    border: none;
}}

.stButton > button:hover {{
    background-color: rgba(0, 0, 0, 0.85);
}}

.result-approve {{
    background: rgba(34,197,94,0.35);
    color: #ecfdf5;
    font-size: 26px;
    font-weight: bold;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}}

.result-reject {{
    background: rgba(239,68,68,0.35);
    color: #fef2f2;
    font-size: 26px;
    font-weight: bold;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
}}

.nav-title {{
    color: #fbbf24;
    font-weight: 700;
}}

.stAlert {{
    background: rgba(15, 23, 42, 0.85) !important;
    color: #f8fafc !important;
    border-radius: 10px;
}}

footer {{
    margin-top: 60px;
    text-align: center;
    color: #e5e7eb;
    font-size: 14px;
}}
/* ===== FIX SELECTBOX DROPDOWN OPTIONS VISIBILITY ===== */

/* Dropdown popup container */
div[data-baseweb="popover"] {{
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    z-index: 9999 !important;
}}

/* Dropdown option list */
ul[role="listbox"] {{
    background-color: #ffffff !important;
    padding: 6px !important;
}}

/* Individual options */
li[role="option"] {{
    color: #111827 !important;
    font-size: 15px !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
}}

/* Hover effect */
li[role="option"]:hover {{
    background-color: #2563eb !important;
    color: #ffffff !important;
}}

/* Selected option */
li[aria-selected="true"] {{
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
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
# SESSION STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "predictions" not in st.session_state:
    st.session_state.predictions = []

# --------------------------------------------------
# TOP NAVBAR
# --------------------------------------------------
col1, col2, col3, col4, col5 = st.columns([5,1,1,1,1])

with col1:
    st.markdown("<h2 class='nav-title'>Loan Predictor</h2>", unsafe_allow_html=True)

with col2:
    if st.button("Home"):
        st.session_state.page = "Home"

with col3:
    if st.button("Predict"):
        st.session_state.page = "Predict"

with col4:
    if st.button("Analytics"):
        st.session_state.page = "Analytics"

with col5:
    if st.button("About"):
        st.session_state.page = "About"

st.markdown("<hr>", unsafe_allow_html=True)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if st.session_state.page == "Home":
    st.markdown("# Loan Approval Prediction System")
    st.markdown("### Intelligent Financial Decision Support using Machine Learning")

    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=230)

    st.markdown("""
The **Loan Approval Prediction System** is a machine learning–based web application
that predicts whether a loan application is **Approved or Rejected** based on
important financial and personal attributes of the applicant.

This system helps financial institutions make **faster, consistent, and data-driven**
loan decisions.

---

### Key Features:
- User-friendly web interface  
- Real-time loan eligibility prediction  
- Probability-based decision output  
- Interactive analytics dashboard  

---

### Parameters Considered:
- Applicant Income  
- Credit Score  
- Loan Amount  
- Employment Type  
- Number of Dependents  
- Loan Term  
- Property Area  

---

### Objective :
To reduce manual effort and improve accuracy in loan approval decisions
using **Machine Learning classification techniques**.

> ⚠️ *This project is developed for educational and academic purposes only.*
""")

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif st.session_state.page == "Predict":
    st.markdown("# Loan Eligibility Check")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Applicant Income", min_value=0)
        credit = st.number_input("Credit Score", 300, 900, 650)
        loan = st.number_input("Loan Amount", min_value=0)
        emp = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])

    with col2:
        dep = st.number_input("Dependents", 0, 10, 0)
        term = st.slider("Loan Term (Months)", 12, 360, 120)
        area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Loan Status"):
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

        st.session_state.predictions.append(pred)

        if pred == 1:
            st.markdown(
                f"<div class='result-approve'>✅ Loan Approved ({prob:.2f}%)</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-reject'>❌ Loan Rejected ({prob:.2f}%)</div>",
                unsafe_allow_html=True
            )

# --------------------------------------------------
# ANALYTICS PAGE
# --------------------------------------------------
elif st.session_state.page == "Analytics":
    st.markdown("# Loan Analytics")

    approvals = st.session_state.predictions.count(1)
    rejections = st.session_state.predictions.count(0)

    if approvals + rejections == 0:
        st.info("No predictions available yet.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.pie(
                [approvals, rejections],
                labels=["Approved", "Rejected"],
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.bar(["Approved", "Rejected"], [approvals, rejections])
            st.pyplot(fig)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif st.session_state.page == "About":
    st.markdown("# About This Project")

    st.markdown("""
The **Loan Approval Prediction System** is a machine learning–powered web application
designed to predict loan eligibility using applicant financial information.

The system uses a **Logistic Regression** algorithm to classify loan applications
as **Approved** or **Rejected** with a confidence score.

---

### Machine Learning Details:
- Algorithm: Logistic Regression  
- Type: Binary Classification  
- Output: Loan Approval Status  
- Probability Score for decision confidence  

---

### Technologies Used:
- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- Matplotlib  

---

### Use Case:
- Academic mini / major project  
- College demonstrations  
- Machine learning portfolio  

---

### Future Enhancements:
- Multiple ML model comparison  
- Database integration  
- User authentication  
- Cloud deployment  

This project highlights the **real-world application of Machine Learning**
in the financial domain.
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<footer>
Loan Approval Prediction App | © 2026
</footer>
""", unsafe_allow_html=True)
