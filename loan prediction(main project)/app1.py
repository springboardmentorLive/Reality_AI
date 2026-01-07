import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import time
import base64

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LoanInsight",
    page_icon="🏦",
    layout="wide"
)

# ================= LOAD LOCAL IMAGE =================
def get_base64_bg(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_base64 = get_base64_bg("bg.jpg")

# ================= CSS =================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.header {{
    background: rgba(20, 40, 50, 0.9);
    padding: 24px 40px;
    border-radius: 0 0 22px 22px;
    margin-bottom: 25px;
}}

.header {{
    display: flex;
    align-items: center;
    gap: 20px;
}}

.header-title {{
    font-size: 44px;
    font-weight: 900;
    color: white;
    margin: 0;
}}

.header-sub {{
    color: #e5e7eb;
    font-size: 18px;
}}

.glass {{
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(18px);
    padding: 26px;
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}}

.section-title {{
    font-size: 36px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 15px;
}}

.metric {{
    background: rgba(255,255,255,0.7);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}}

label, p, li, h3, span {{
    color: #0f172a !important;
    font-size: 16px;
}}

/* ================= NAVIGATION TABS - INCREASED SIZE & SPACING ================= */
div[data-testid="stTab"] > div > div {{
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    margin: 0 8px !important;
}}

div[data-testid="stTab"] button {{
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    margin: 0 12px !important;
    height: 50px !important;
}}

div[data-testid="stTab"] button[aria-selected="true"] {{
    font-size: 22px !important;
    padding: 14px 30px !important;
}}

/* ================= FIX VISIBILITY ISSUES ================= */

/* Slider numbers (min, max, current value) */
div[data-testid="stSlider"] * {{
    color: black !important;
    font-weight: 600;
    font-size: 16px !important;
}}

/* Selectbox closed */
div[data-baseweb="select"] > div {{
    background-color: white !important;
    color: black !important;
}}

div[data-baseweb="select"] span {{
    color: black !important;
    font-weight: 600;
    font-size: 18px !important;
}}

/* Selectbox dropdown (open) */
div[data-baseweb="popover"],
ul[role="listbox"] {{
    background-color: white !important;
}}

ul[role="listbox"] li {{
    background-color: white !important;
    color: black !important;
    font-weight: 600;
    font-size: 20px !important;
    padding: 14px 18px !important;
}}

ul[role="listbox"] li:hover {{
    background-color: #e5e7eb !important;
    color: black !important;
}}

/* Radio buttons - Increased font size */
div[data-testid="stRadio"] label {{
    font-size: 20px !important;
    font-weight: 700 !important;
}}

div[data-testid="stRadio"] input + label {{
    font-size: 20px !important;
    font-weight: 700 !important;
}}

/* Predict button */
div.stButton > button {{
    background-color: white !important;
    color: black !important;
    font-weight: 700;
    border-radius: 10px;
    border: 2px solid black !important;
    font-size: 20px !important;
    padding: 14px 28px !important;
}}

div.stButton > button:hover {{
    background-color: #e5e7eb !important;
    color: black !important;
}}
</style>
""", unsafe_allow_html=True)

# ================= DATA =================
np.random.seed(42)
df = pd.DataFrame({
    "Income": np.random.randint(20000, 150000, 120),
    "CreditScore": np.random.randint(300, 850, 120),
    "LoanAmount": np.random.randint(10000, 200000, 120),
    "EmploymentType": np.random.choice([0, 1], 120),
    "LoanTenure": np.random.choice([1, 3, 5, 7], 120),
    "Age": np.random.randint(22, 65, 120)
})

df["Approved"] = (
    (df["Income"] > 40000) &
    (df["CreditScore"] > 600) &
    (df["LoanAmount"] < 150000) &
    (df["Age"] < 60)
).astype(int)

# ================= MODEL =================
X = df[["Income", "CreditScore", "LoanAmount", "EmploymentType", "LoanTenure", "Age"]]
y = df["Approved"]
model = LogisticRegression()
model.fit(X, y)

# ================= HEADER =================
st.markdown("""
<div class="header">
    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135706.png" width="60" height="60" alt="Bank Icon">
    <div>
        <div class="header-title">LoanInsight</div>
        <div class="header-sub">Smart Loan Approval Prediction System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
home, predict, analytics, dataset, about = st.tabs(
    ["🏠 Home", "📊 Predict", "📈 Analytics", "📁 Dataset", "ℹ️ About"]
)

# ================= HOME =================
with home:
    st.markdown("<div class='section-title'>Dashboard Overview</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass">
            <h3>📌 What is LoanInsight?</h3>
            <p>
            <b>LoanInsight</b> is an intelligent loan approval prediction system
            that uses <b>Machine Learning</b> to estimate whether a customer is
            eligible for a loan.
            </p>
            <p>
            Banks receive thousands of loan applications daily. Manual evaluation
            is time-consuming and inconsistent. LoanInsight helps automate
            decision-making using data-driven logic.
            </p>
            <p><b>Input Parameters</b></p>
            <ul>
                <li>Monthly Income</li>
                <li>Credit Score</li>
                <li>Requested Loan Amount</li>
                <li>Employment Status</li>
                <li>Loan Tenure</li>
                <li>Age</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass">
            <h3>🎯 Why Use LoanInsight?</h3>
            <ul>
                <li>Reduces human bias in approvals</li>
                <li>Provides fast & consistent decisions</li>
                <li>Helps banks assess risk efficiently</li>
                <li>Improves customer experience</li>
            </ul>
            <p>
            This system is designed for <b>educational and demonstration</b>
            purposes to understand ML in finance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"<div class='metric'><h3>{len(df)}</h3><p>Total Applications</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><h3>{df['Approved'].sum()}</h3><p>Approved Loans</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><h3>{df['Approved'].mean()*100:.1f}%</h3><p>Approval Rate</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric'><h3>{int(df['CreditScore'].mean())}</h3><p>Avg Credit Score</p></div>", unsafe_allow_html=True)

# ================= PREDICT =================
with predict:
    st.markdown("<div class='section-title'>Loan Approval Prediction</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <h3>📋 Eligibility Rules Used by the Model</h3>
        <ul>
            <li>Income should generally be above ₹40,000</li>
            <li>Credit score above 600 increases approval chances</li>
            <li>Loan amount should be within a reasonable limit</li>
            <li>Employed applicants have higher approval probability</li>
            <li>Shorter loan tenures may improve approval</li>
            <li>Age under 60 preferred</li>
        </ul>
        <p>
        These rules are learned automatically by the Logistic Regression model
        from historical data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("💰 Monthly Income", 10000, 200000, 50000)
        credit = st.slider("📊 Credit Score", 300, 850, 650)
        age = st.slider("👤 Age", 18, 70, 35)

    with col2:
        loan = st.slider("🏦 Loan Amount", 5000, 300000, 100000)
        emp = st.radio("👔 Employment Type",["Unemployed", "Employed"],horizontal=True)
        tenure = st.radio("📅 Loan Tenure (Years)",["1", "3", "5", "7"],horizontal=True)

    if st.button("🚀 Predict Loan"):
        emp_val = 1 if emp == "Employed" else 0
        tenure_val = {"1":1, "3":3, "5":5, "7":7}[tenure]
        with st.spinner("Analyzing profile..."):
            time.sleep(1)
        confidence = model.predict_proba([[income, credit, loan, emp_val, tenure_val, age]])[0][1] * 100

        if confidence >= 50:
            st.success(f"✅ Loan Approved — Confidence {confidence:.2f}%")
        else:
            st.error(f"❌ Loan Rejected — Confidence {confidence:.2f}%")

# ================= ANALYTICS =================
with analytics:
    st.markdown("<div class='section-title'>Analytics & Insights</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <h3>🔍 What Are We Analyzing?</h3>
        <ul>
            <li>Loan approval distribution</li>
            <li>Relationship between income and credit score</li>
            <li>Risk patterns in applicant profiles</li>
        </ul>
        <p>
        These analytics help banks understand approval trends and improve
        lending strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df["Approved"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xticklabels(["Rejected", "Approved"], rotation=0)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.scatter(df["Income"], df["CreditScore"], c=df["Approved"])
        ax.set_xlabel("Income")
        ax.set_ylabel("Credit Score")
        st.pyplot(fig)

# ================= DATASET =================
with dataset:
    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
    st.dataframe(df)

# ================= ABOUT =================
with about:
    st.markdown("<div class='section-title'>About LoanInsight</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <p>
        <b>LoanInsight</b> is a machine learning-based application built using
        <b>Python, Streamlit, Pandas, NumPy, and Scikit-learn</b>.
        </p>
        <p>
        The system uses <b>Logistic Regression</b>, a supervised learning algorithm,
        to predict loan approval probability.
        </p>
        <h3>📌 Key Highlights</h3>
        <ul>
            <li>Glassmorphism modern UI</li>
            <li>Realistic financial features</li>
            <li>Probability-based prediction</li>
            <li>Educational finance use case</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
