import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os

# -----------------------------
# Load dataset
# -----------------------------
if not os.path.exists("loan_data.csv"):
    raise FileNotFoundError("loan_data.csv not found! Make sure your dataset is in the project folder.")

data = pd.read_csv("loan_data.csv")

# -----------------------------
# Check and fix column names
# -----------------------------
# Make column names uniform (remove spaces)
data.columns = [col.strip().replace(" ", "") for col in data.columns]

# -----------------------------
# Encode categorical features
# -----------------------------
le_emp = LabelEncoder()
data['EmploymentType'] = le_emp.fit_transform(data['EmploymentType'])

le_area = LabelEncoder()
data['PropertyArea'] = le_area.fit_transform(data['PropertyArea'])

# -----------------------------
# Features & target
# -----------------------------
X = data[['Income', 'CreditScore', 'LoanAmount', 'EmploymentType', 'Dependents', 'LoanTerm', 'PropertyArea']]
y = data['Loan_Status']

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Save model and encoders
# -----------------------------
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(le_emp, open("employment_encoder.pkl", "wb"))
pickle.dump(le_area, open("area_encoder.pkl", "wb"))

print("Model and encoders trained and saved successfully!")
