import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("loan_data.csv")

# Encode categorical columns
le_emp = LabelEncoder()
le_area = LabelEncoder()

df["EmploymentType"] = le_emp.fit_transform(df["EmploymentType"])
df["PropertyArea"] = le_area.fit_transform(df["PropertyArea"])

# Features & target
X = df[
    [
        "Income",
        "CreditScore",
        "LoanAmount",
        "EmploymentType",
        "Dependents",
        "LoanTerm",
        "PropertyArea"
    ]
]
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & encoders
pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(le_emp, open("employment_encoder.pkl", "wb"))
pickle.dump(le_area, open("area_encoder.pkl", "wb"))

print("✅ Model trained & saved successfully")
