import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fbff, #eef3f8);
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 46px;
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
    background: white;
    padding: 25px;
    border-radius: 16px;
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
# Load Model
# --------------------------------------------------
model = pickle.load(open("loan_model.pkl", "rb"))
le_emp = pickle.load(open("employment_encoder.pkl", "rb"))
le_area = pickle.load(open("area_encoder.pkl", "rb"))

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "📊 Loan Prediction", "ℹ️ About"])

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>Loan Approval Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI-powered decision support system</p>", unsafe_allow_html=True)

    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

    st.markdown("""
    <div class="card">
        <h3>About This Application</h3>
        <p>
        This application predicts whether a loan will be approved or rejected
        using Machine Learning.
        </p>
        <ul>
            <li>Applicant Income</li>
            <li>Credit Score</li>
            <li>Loan Amount</li>
            <li>Employment Type</li>
            <li>Property Area</li>
        </ul>
        <b>Disclaimer:</b> Educational use only.
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
elif page == "📊 Loan Prediction":
    st.markdown("<h1 class='main-title'>Loan Eligibility Check</h1>", unsafe_allow_html=True)

    income = st.number_input("Income", min_value=0)
    credit = st.number_input("Credit Score", 300, 900, 650)
    loan = st.number_input("Loan Amount", min_value=0)
    emp = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
    dep = st.number_input("Dependents", 0, 10, 0)
    term = st.slider("Loan Term (Months)", 12, 360, 120)
    area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict"):
        emp = le_emp.transform([emp])[0]
        area = le_area.transform([area])[0]

        data = pd.DataFrame([[income, credit, loan, emp, dep, term, area]],
                            columns=["Income","Credit","Loan","Employment","Dependents","Term","Area"])

        pred = model.predict(data)[0]
        prob = model.predict_proba(data).max()*100

        if pred == 1:
            st.markdown(f"<div class='result-approve'>Loan Approved ({prob:.2f}%)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-reject'>Loan Rejected ({prob:.2f}%)</div>", unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT (NO CODE DISPLAY)
# --------------------------------------------------
elif page == "ℹ️ About":
    st.header("ℹ️ About this App")
    st.write("""
    **Loan Approval Prediction App** predicts whether a loan will be approved based on applicant details.

    **Features:**
    - Input Income, Credit Score, Loan Amount, Employment Type
    - Predict Approval Status with Confidence %
    - Educational project using ML

    **Tech Stack:**
    - Streamlit for UI
    - Scikit-learn for ML (Logistic Regression / Decision Tree)

    ⚠️ For educational purposes only.
    """)
