import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import os

st.title("Predict Student Performance")

# -------------------------
# Ensure folders exist
# -------------------------
os.makedirs("models", exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
if not os.path.exists("data/student_data.csv"):
    st.error("Dataset not found! Place 'student_data.csv' inside the data folder.")
    st.stop()

data = pd.read_csv("data/student_data.csv")

# -------------------------
# Features and targets
# -------------------------
X = data[['study_hours', 'attendance', 'internal_marks']]
y_marks = data['final_marks']
y_pass = (y_marks >= 35).astype(int)  # 1 = Pass, 0 = Fail

# -------------------------
# Safety check
# -------------------------
if y_pass.nunique() < 2:
    st.error("Pass/Fail column must contain both Pass (1) and Fail (0) samples.")
    st.stop()

# -------------------------
# Train-test split (ONE split, stratified)
# -------------------------
X_train, X_test, y_marks_train, y_marks_test, y_pass_train, y_pass_test = train_test_split(
    X,
    y_marks,
    y_pass,
    test_size=0.2,
    random_state=42,
    stratify=y_pass
)

# -------------------------
# Train models
# -------------------------
marks_model = LinearRegression()
marks_model.fit(X_train, y_marks_train)
joblib.dump(marks_model, "models/marks_model.pkl")

pass_model = LogisticRegression(max_iter=1000)
pass_model.fit(X_train, y_pass_train)
joblib.dump(pass_model, "models/pass_model.pkl")

# -------------------------
# User Input
# -------------------------
st.header("Enter Student Details")

study_hours = st.slider("Study Hours per Day", 0.0, 10.0, 4.0, 0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal_marks = st.slider("Internal Marks", 0, 50, 25)

X_input = np.array([[study_hours, attendance, internal_marks]])

# -------------------------
# Predictions
# -------------------------
predicted_marks = marks_model.predict(X_input)[0]
pass_prob = pass_model.predict_proba(X_input)[0][1]
pass_fail = pass_model.predict(X_input)[0]

# -------------------------
# Output
# -------------------------
st.subheader("Prediction Results")
st.write(f"Predicted Marks: {predicted_marks:.2f}")
st.write(f"Pass Probability: {pass_prob * 100:.2f}%")
st.write(f"Predicted Result: {'Pass' if pass_fail == 1 else 'Fail'}")
