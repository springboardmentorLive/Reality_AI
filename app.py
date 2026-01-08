import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Performance App",
    page_icon="🎓",
    layout="wide"
)

# ---------------- TOP NAVIGATION ----------------
tabs = st.tabs([
    " Home",
    " Prediction",
    " Analytics",
    " About",
    " Contact"
])

# ================= HOME =================
with tabs[0]:
    st.markdown("##  Student Performance Prediction System")
    st.write("""
    This application predicts student academic performance using
    Machine Learning based on study habits and attendance.
    """)
    st.info("Use the Prediction tab to get started.")

# ================= PREDICTION =================
with tabs[1]:
    st.markdown("##  Enter Student Details")

    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student Name")
    with col2:
        roll_number = st.text_input("Roll Number")

    c1, c2, c3 = st.columns(3)
    with c1:
        study_hours = st.slider("Study Hours / Day", 0.0, 10.0, 4.0, 0.5)
    with c2:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
    with c3:
        internal_marks = st.slider("Internal Marks", 0, 50, 25)

    model = joblib.load("models/marks_model.pkl")

    X = np.array([[study_hours, attendance, internal_marks]])
    predicted_marks = model.predict(X)[0]

    st.markdown("##  Prediction Result")

    if predicted_marks >= 35:
        st.success(f"""
        **PASS**  
        Student: {student_name}  
        Roll No: {roll_number}  
        Predicted Marks: {predicted_marks:.2f}
        """)
        status = "Pass"
    else:
        st.error(f"""
        **FAIL**  
        Student: {student_name}  
        Roll No: {roll_number}  
        Predicted Marks: {predicted_marks:.2f}
        """)
        status = "Fail"

# ================= ANALYTICS =================
with tabs[2]:
    st.markdown("##  Visual Analysis")

    col1, col2 = st.columns(2)

    # Bar Chart
    with col1:
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        ax1.bar(["Marks"], [predicted_marks], color="green")
        ax1.set_ylim(0, 100)
        ax1.set_title("Predicted Marks")
        st.pyplot(fig1)

   

# ================= ABOUT =================
with tabs[3]:
    st.markdown("##  About This Project")
    st.write("""
    **Student Performance Prediction System** is a Machine Learning based
    web application developed using **Python, Scikit-Learn, and Streamlit**.

    ### Features:
    - Predicts final marks
    - Determines Pass/Fail status
    - Simple & interactive UI
    - Visual analytics support

    ### Use Case:
    Helps teachers and students analyze academic performance early.
    """)

# ================= CONTACT =================
with tabs[4]:
    st.markdown("## 📞 Contact")
    st.write("""
    **Developer:** Student  
    **Technology:** Python, Machine Learning, Streamlit  
    **Purpose:** Academic Project
    """)
