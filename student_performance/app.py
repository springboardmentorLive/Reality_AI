import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------- LIGHTENED BACKGROUND IMAGE ----------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(
                rgba(255,255,255,0.75),
                rgba(255,255,255,0.75)
            ),
            url("https://images.unsplash.com/photo-1524995997946-a1c2e315a42f");
        background-size: cover;
        background-attachment: fixed;
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: black !important;
    }

    input, textarea {
        background-color: white !important;
        color: black !important;
        border: 1px solid black !important;
    }

    button {
        background-color: #1f4ed8 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ------------------------------------------------------------

# ---------------- LOAD DATA ----------------
data = pd.read_csv("student_data.csv")

X = data[['study_hours', 'attendance', 'internal_marks']]
y = data['pass']

model = LogisticRegression()
model.fit(X, y)

# ---------------- NAVIGATION BAR ----------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎓 Predict Result", "📊 Subject-wise Marks"]
)

# ---------------- HOME PAGE ----------------
if menu == "🏠 Home":
    st.title("🎓 Student Performance Prediction App")
    st.write("""
    This application predicts student performance and visualizes
    **subject-wise marks** using Machine Learning.
    """)

# ---------------- PREDICTION PAGE ----------------
elif menu == "🎓 Predict Result":
    st.title("🎯 Predict Student Result")

    study_hours = st.slider("📘 Study Hours per Day", 0, 10, 3)
    attendance = st.slider("📊 Attendance (%)", 0, 100, 60)
    internal_marks = st.slider("📝 Average Internal Marks", 0, 100, 50)

    if st.button("🔍 Predict"):
        prediction = model.predict([[study_hours, attendance, internal_marks]])[0]

        if prediction == 1:
            st.success("✅ Student is likely to PASS")
        else:
            st.error("❌ Student is likely to FAIL")

# ---------------- SUBJECT-WISE GRAPH PAGE ----------------
elif menu == "📊 Subject-wise Marks":
    st.title("📊 Subject-wise Marks Analysis")

    maths = st.slider("📐 Mathematics", 0, 100, 60)
    science = st.slider("🔬 Science", 0, 100, 55)
    english = st.slider("📘 English", 0, 100, 65)
    computer = st.slider("💻 Computer Science", 0, 100, 70)

    marks_data = pd.DataFrame(
        {"Marks": [maths, science, english, computer]},
        index=["Maths", "Science", "English", "Computer"]
    )

    st.subheader("📈 Subject-wise Marks Graph")
    st.bar_chart(marks_data)

    avg_marks = (maths + science + english + computer) / 4
    st.subheader("📊 Overall Performance")
    st.progress(avg_marks / 100)
    st.write(f"**Average Marks:** {avg_marks:.2f}")

    if avg_marks >= 40:
        st.success("✔ Overall Passing Performance")
    else:
        st.warning("⚠ Needs Improvement")
