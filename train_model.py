import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample training data (example)
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6],
    "marks": [35, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

X = df[["hours_studied"]]
y = df["marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/marks_model.pkl")

print("marks_model.pkl created successfully")
