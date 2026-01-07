import streamlit as st
import pandas as pd
import time
import base64

# =========================
# FUNCTION TO LOAD IMAGE
# =========================
def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

main_bg = get_base64("OIP (4).jpg")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Diabetes Risk Dashboard",
    page_icon="❤️",
    layout="wide"
)

# =========================
# CSS STYLING
# =========================
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{main_bg}");
    background-size: cover;
    background-attachment: fixed;
}}

.navbar {{
    background: linear-gradient(to right,#2563EB,#4F46E5);
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 1rem;
}}

.card {{
    background: rgba(255,255,255,0.95);
    padding: 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    color: #111827;
}}

div[role="radiogroup"] {{
    display: flex;
    justify-content: center;
    gap: 25px;
}}

div[role="radiogroup"] > label {{
    background: white;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    cursor: pointer;
}}

div[role="radiogroup"] > label:hover {{
    background: #E0E7FF;
}}

h1,h2,h3,h4,p {{
    color: #111827 !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# =========================
# PREDICTION FUNCTION
# =========================
def predict_diabetes(age, bmi, bp, glucose):
    score = 0
    if age > 45: score += 1
    if bmi > 25: score += 1
    if bmi > 30: score += 1
    if bp > 80: score += 1
    if bp > 90: score += 1
    if glucose > 100: score += 2
    if glucose > 125: score += 3

    confidence = min(score * 10, 95)

    if score >= 6:
        return "High", confidence
    elif score >= 3:
        return "Moderate", confidence
    else:
        return "Low", confidence

# =========================
# HEADER + TOP NAVBAR
# =========================
st.markdown("""
<div class="navbar">
    <h1 style="text-align:center; color:white;">❤️ Diabetes Risk Prediction System</h1>
    <p style="text-align:center; color:white;">Healthcare Analytics & Prevention</p>
</div>
""", unsafe_allow_html=True)

page = st.radio(
    "",
    ["Home", "Dashboard", "Healthy Habits", "Diet Plan", "Prevention Tips"],
    horizontal=True
)

st.markdown("---")


st.markdown("""
<style>
/* Targets the button within the Streamlit app */
div.stButton > button:first-child { Background-color: #2563EB; /* Sets the background color */
    color: red; /* Sets the text color to red */
}
}
</style>
""", unsafe_allow_html=True)

# =========================
# HOME PAGE
# =========================
if page == "Home":

    st.subheader("🧑 Patient Details")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.slider("Age", 18, 90, 45)
    with c2:
        bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    with c3:
        bp = st.slider("Blood Pressure", 40, 130, 80)
    with c4:
        glucose = st.slider("Glucose Level", 40, 200, 100)

    if st.button("🔍 Predict Diabetes Risk"):
        time.sleep(0.5)
        st.session_state.predicted = True
        st.session_state.age = age
        st.session_state.bmi = bmi
        st.session_state.bp = bp
        st.session_state.glucose = glucose

        risk, confidence = predict_diabetes(age, bmi, bp, glucose)

        # ✅ Styled Result Cards
        st.markdown(f"""
        <div style="background-color:#16A34A;padding:15px;border-radius:12px;
                    border-left:6px solid #16A34A;font-size:18px;font-weight:600;">
            ✅ Risk Level: {risk}
        </div>

        <div style="background-color:#2563EB;padding:15px;border-radius:12px;
                    border-left:6px solid #2563EB;font-size:18px;font-weight:600;margin-top:10px;">
            ℹ️ Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)

# =========================
# DASHBOARD PAGE
# =========================
elif page == "Dashboard":

    if not st.session_state.predicted:
        st.warning("⚠️ Please predict first from Home page.")
    else:
        st.subheader("📊 Health Analytics Dashboard")

        # BAR CHART DATA
        health_df = pd.DataFrame({
            "Metric": ["Age", "BMI", "Blood Pressure", "Glucose"],
            "Value": [
                st.session_state.age,
                st.session_state.bmi,
                st.session_state.bp,
                st.session_state.glucose
            ]
        })

        st.markdown("<style = 'background-color:#16A34A;'></style>Patient Health Metrics", unsafe_allow_html=True)
        st.bar_chart(health_df.set_index("Metric"))

        # LINE GRAPH DATA
        risk_df = pd.DataFrame({
            "Factor": ["Age", "BMI", "Blood Pressure", "Glucose"],
            "Risk Score": [
                1 if st.session_state.age > 45 else 0,
                2 if st.session_state.bmi > 30 else 1 if st.session_state.bmi > 25 else 0,
                2 if st.session_state.bp > 90 else 1 if st.session_state.bp > 80 else 0,
                3 if st.session_state.glucose > 125 else 2 if st.session_state.glucose > 100 else 0
            ]
        })

        st.markdown("<style = 'background-color:#2563EB;'></style>Risk Contribution Line Graph", unsafe_allow_html=True)
        st.line_chart(risk_df.set_index("Factor"))

# =========================
# HEALTHY HABITS
# =========================
elif page == "Healthy Habits":

    st.subheader("🏃 Healthy Lifestyle Tips")
    habits = [
        "🚶 Walk 30 minutes daily",
        "💧 Drink enough water",
        "🏋️ Regular exercise",
        "🛌 Proper sleep",
        "🚭 Avoid smoking & alcohol"
    ]
    for h in habits:
        st.markdown(f"<div class='card'>{h}</div>", unsafe_allow_html=True)

# =========================
# DIET PLAN
# =========================
elif page == "Diet Plan":

    st.subheader("🥗 Recommended Diet Plan")
    diet = {
        "Breakfast": "Oats, fruits, eggs",
        "Lunch": "Brown rice, vegetables",
        "Snacks": "Fruits & nuts",
        "Dinner": "Light meals",
        "Avoid": "Sugary & junk food"
    }

    for k, v in diet.items():
        st.markdown(f"<div class='card'><b>{k}:</b> {v}</div>", unsafe_allow_html=True)

# =========================
# PREVENTION TIPS
# =========================
elif page == "Prevention Tips":

    st.subheader("🚫 Diabetes Prevention Tips")
    tips = [
        "🩺 Regular checkups",
        "🥦 Fiber-rich food",
        "📉 Maintain healthy weight",
        "🧘 Stress control",
        "📵 Avoid sedentary lifestyle"
    ]

    for t in tips:
        st.markdown(f"<div class='card'>{t}</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center><b>Diabetes Risk Prediction System</b> | Streamlit Healthcare Project</center>",
    unsafe_allow_html=True
)
