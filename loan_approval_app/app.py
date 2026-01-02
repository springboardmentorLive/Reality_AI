import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS (ONLY for layout, not text content)
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    color: #1f77b4;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.result-approve {
    color: green;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
.result-reject {
    color: red;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Encoders
# --------------------------------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
le_emp = pickle.load(open("employment_encoder.pkl", "rb"))
le_area = pickle.load(open("area_encoder.pkl", "rb"))

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Loan Prediction", "About"])

# --------------------------------------------------
# HOME PAGE (PURE MATTER)
# --------------------------------------------------
if page == "Home":
    st.markdown("<div class='main-title'>Loan Approval Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>AI-powered loan eligibility assessment system</div>", unsafe_allow_html=True)

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
# LOAN PREDICTION PAGE
# --------------------------------------------------
elif page == "Loan Prediction":
    st.markdown("<div class='main-title'>Loan Eligibility Check</div>", unsafe_allow_html=True)

    income = st.number_input("Income", min_value=0)
    credit = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    loan = st.number_input("Loan Amount", min_value=0)
    emp = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
    dep = st.number_input("Dependents", 0, 10, 0)
    term = st.slider("Loan Term (Months)", 12, 360, 120)
    area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict"):
        emp_encoded = le_emp.transform([emp])[0]
        area_encoded = le_area.transform([area])[0]

        data = pd.DataFrame(
            [[income, credit, loan, emp_encoded, dep, term, area_encoded]],
            columns=[
                "Income",
                "CreditScore",
                "LoanAmount",
                "EmploymentType",
                "Dependents",
                "LoanTerm",
                "PropertyArea"
            ]
        )

        pred = model.predict(data)[0]
        prob = model.predict_proba(data).max() * 100

        if pred == 1:
            st.markdown(f"<div class='result-approve'>Loan Approved ({prob:.2f}%)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-reject'>Loan Rejected ({prob:.2f}%)</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT PAGE (PURE MATTER)
# --------------------------------------------------
elif page == "About":
    st.markdown("<div class='main-title'>About This Project</div>", unsafe_allow_html=True)

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
