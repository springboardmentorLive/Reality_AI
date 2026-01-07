import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from fpdf import FPDF
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Prediction System", layout="wide")

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "user" not in st.session_state:
    st.session_state.user = "Guest"
if "users" not in st.session_state:
    st.session_state.users = {}
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- BACKGROUND ----------------
def set_bg(img):
    with open(img, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:url("data:image/jpg;base64,{encoded}");
            background-size:cover;
            background-attachment:fixed;
        }}
        .stButton>button {{
            color:black;
            font-weight:bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- LOAD DATA ----------------
df = pd.read_csv("loan_data.csv")
emp_map = {"Unemployed":0,"Self-Employed":1,"Salaried":2}
df["Employment"] = df["Employment"].map(emp_map)

X = df[["Income","Credit Score","Loan Amount","Employment","Age","Existing Loans"]]
y = df["Approved"]

model = LogisticRegression(max_iter=500)
model.fit(X,y)

# ---------------- PREDICT ----------------
def predict_loan(data):
    pred = model.predict(data)[0]
    prob = max(model.predict_proba(data)[0])*100
    if pred==1:
        reason = "Good credit score, sufficient income and low existing loans."
    else:
        reason = "Low credit score, insufficient income, or too many existing loans."
    return pred, prob, reason

# ---------------- EMI CALCULATION ----------------
def calculate_emi(principal, rate, months):
    if rate==0:
        return principal/months
    monthly_rate = rate/(12*100)
    emi = principal * monthly_rate * ((1+monthly_rate)**months)/(((1+monthly_rate)**months)-1)
    return emi

# ---------------- PDF REPORT ----------------
def generate_pdf(result, confidence, inputs, emi):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","",14)
    pdf.cell(0,10,"Loan Prediction Report",ln=True)
    pdf.ln(10)
    status = "Approved" if result==1 else "Rejected"
    pdf.cell(0,10,f"Result: {status}",ln=True)
    pdf.cell(0,10,f"Confidence: {confidence:.2f}%",ln=True)
    pdf.ln(5)
    pdf.cell(0,10,"Input Details:",ln=True)
    for k,v in inputs.items():
        pdf.cell(0,10,f"{k}: {v}",ln=True)
    pdf.ln(5)
    pdf.cell(0,10,f"Calculated EMI: ₹{emi:.2f}",ln=True)
    pdf.output("loan_report.pdf","F")

# ---------------- NAV BAR ----------------
nav = st.columns(9)
labels = ["Home","Eligibility","My Predictions","Dashboard","Chatbot","Signup","Login","About","Logout"]
for i,l in enumerate(labels):
    if nav[i].button(l, key=f"nav_btn_{i}"):
        st.session_state.page = l

# ---------------- HOME ----------------
if st.session_state.page == "Home":
    set_bg("images/image1.jpg")
    st.markdown("<h2 style='color:black;'>🏦 Personal Loan Prediction System</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:black;'>Smart • Secure • Machine Learning Powered</h4>", unsafe_allow_html=True)

    imgs = ["images/image1.jpg","images/image2.jpg","images/image3.jpg"]
    img_slot = st.empty()
    for img in imgs:
        img_slot.image(img, width=900)
        time.sleep(1.2)

# ---------------- ELIGIBILITY ----------------
elif st.session_state.page == "Eligibility":
    set_bg("images/image2.jpg")
    st.markdown("<h3 style='color:black;'>✅ Check Loan Eligibility</h3>", unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        income = st.number_input("Income",0,key="income_input")
        credit = st.number_input("Credit Score",300,850,key="credit_input")
        age = st.number_input("Age",18,key="age_input")
    with col2:
        loan = st.number_input("Loan Amount",0,key="loan_input")
        emp = st.selectbox("Employment",list(emp_map.keys()), key="emp_input")
        existing = st.number_input("Existing Loans",0,key="existing_input")
        rate = st.number_input("Interest Rate (%)",0.0,50.0,10.0,key="rate_input")
        term = st.number_input("Loan Term (Months)",6,360,12,key="term_input")

    if st.button("Predict", key="predict_btn"):
        data = np.array([[income,credit,loan,emp_map[emp],age,existing]])
        result, conf, reason = predict_loan(data)
        emi = calculate_emi(loan, rate, term)
        status = "Approved" if result==1 else "Rejected"

        # Polished card
        st.markdown(f"""
        <div style='background-color:white; border-radius:15px; padding:15px; margin:10px; box-shadow:2px 2px 10px grey'>
        <h4 style='color:black;'>Status: {status} ({conf:.2f}%)</h4>
        <p style='color:black;'>Reason: {reason}</p>
        <p style='color:black;'>Calculated EMI: ₹{emi:.2f} per month for {term} months at {rate}% interest.</p>
        </div>
        """, unsafe_allow_html=True)

        # Save user-wise history
        st.session_state.history.append({
            "User": st.session_state.user,
            "Income": income,
            "Credit Score": credit,
            "Loan Amount": loan,
            "Employment": emp,
            "Age": age,
            "Existing Loans": existing,
            "Interest Rate": rate,
            "Term": term,
            "EMI": emi,
            "Status": status,
            "Confidence": conf
        })

        # Generate PDF
        inputs = {"Income":income,"Credit Score":credit,"Loan Amount":loan,"Employment":emp,"Age":age,"Existing Loans":existing,"Interest Rate":rate,"Term":term}
        generate_pdf(result, conf, inputs, emi)

        # Download PDF
        with open("loan_report.pdf","rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="loan_report.pdf", key="pdf_dl_btn")

# ---------------- MY PREDICTIONS ----------------
elif st.session_state.page == "My Predictions":
    set_bg("images/image1.jpg")
    st.markdown("<h3 style='color:black;'>📄 My Predictions</h3>", unsafe_allow_html=True)

    if st.session_state.user=="Guest":
        st.info("Please login to see your predictions.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        user_df = hist_df[hist_df['User']==st.session_state.user]
        if not user_df.empty:
            st.dataframe(user_df)
            fig2 = px.bar(user_df, x="Income", y="EMI", color="Status", title="EMI vs Income")
            st.plotly_chart(fig2)

            # CSV download
            csv_buffer = io.StringIO()
            user_df.to_csv(csv_buffer,index=False)
            st.download_button("📥 Download My Predictions CSV", csv_buffer.getvalue(), file_name="my_predictions.csv")
        else:
            st.info("No predictions yet.")

# ---------------- DASHBOARD ----------------
elif st.session_state.page == "Dashboard":
    if st.session_state.user!="admin":
        st.warning("Admin only")
    else:
        set_bg("images/image1.jpg")
        st.markdown("<h3 style='color:black;'>📊 Admin Dashboard</h3>", unsafe_allow_html=True)

        preds = model.predict(X)
        acc = accuracy_score(y,preds)
        cm = confusion_matrix(y,preds)

        st.success(f"Model Accuracy: {acc*100:.2f}%")

        fig = px.imshow(cm,text_auto=True,
                        labels=dict(x="Predicted",y="Actual"),
                        title="Confusion Matrix")
        st.plotly_chart(fig)
        st.dataframe(df)

# ---------------- SIGNUP ----------------
elif st.session_state.page == "Signup":
    set_bg("images/image3.jpg")
    st.markdown("<h3 style='color:black;'>👤 User Signup</h3>", unsafe_allow_html=True)

    u = st.text_input("Username", key="signup_user")
    p = st.text_input("Password",type="password", key="signup_pass")
    if st.button("Create Account", key="signup_btn"):
        if u and p:
            st.session_state.users[u]=p
            st.success("Account created")
        else:
            st.error("Enter username and password")

# ---------------- LOGIN ----------------
elif st.session_state.page == "Login":
    set_bg("images/image3.jpg")
    st.markdown("<h3 style='color:black;'>🔐 Login</h3>", unsafe_allow_html=True)

    u = st.text_input("Username", key="login_user")
    p = st.text_input("Password",type="password", key="login_pass")
    if st.button("Login", key="login_btn"):
        if u=="admin" and p=="admin123":
            st.session_state.user="admin"
            st.success("Admin logged in")
        elif u in st.session_state.users and st.session_state.users[u]==p:
            st.session_state.user=u
            st.success("User logged in")
        else:
            st.error("Invalid login")

# ---------------- CHATBOT ----------------
elif st.session_state.page == "Chatbot":
    set_bg("images/image3.jpg")
    st.markdown("<h3 style='color:black;'>💬 Chatbot</h3>", unsafe_allow_html=True)

    faq_qa = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! Need help with loans?",
        "how are you": "I'm just a program, but thanks for asking! How can I assist?",
        "what is your name": "I am LoanBot, your assistant.",
        "what is a personal loan?": "A personal loan is an unsecured loan based on your income, credit score, and other criteria.",
        "how is loan eligibility calculated?": "Eligibility is based on income, credit score, age, employment type, existing loans, and requested loan amount.",
        "what is EMI?": "EMI is Equated Monthly Installment, the fixed monthly payment for your loan.",
        "can I check my loan approval instantly?": "Yes! Enter your details in the Eligibility section, and the system predicts instantly.",
        "how can I download my report?": "After prediction, you can download the PDF report from the Eligibility page.",
        "what interest rate should I use?": "Use the rate offered by your bank or any estimated market rate for EMI calculation.",
        "can I save my prediction history?": "Yes! Every prediction is saved under your username and can be viewed in My Predictions (or Dashboard for admin)."
    }

    user_q = st.text_input("Ask a question:", key="faq_input")
    if st.button("Get Answer", key="faq_btn"):
        found = False
        for q,a in faq_qa.items():
            if user_q.lower() in q.lower():
                st.markdown(f"""
                <div style="background-color:#d1e7dd; border-radius:15px; padding:10px; margin:5px; width:fit-content;">
                {a}
                </div>
                """, unsafe_allow_html=True)
                found = True
                break
        if not found:
            st.warning("Sorry, I don't have an answer for that. Try asking something else.")

# ---------------- ABOUT ----------------
elif st.session_state.page == "About":
    set_bg("images/image1.jpg")
    st.markdown("""
    ## ℹ About Project
    This system predicts personal loan approval using Machine Learning.
    Built using Streamlit, Python & Logistic Regression.
    """)

# ---------------- LOGOUT ----------------
elif st.session_state.page == "Logout":
    set_bg("images/image4.jpg")  # or image5.jpg
    st.markdown("<h3 style='color:black;'>👋 Logged Out</h3>", unsafe_allow_html=True)
    st.session_state.user="Guest"
    st.session_state.page="Home"
    st.success("You have been logged out.")
