import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("house_rent.csv")

# Encode categorical columns
le_location = LabelEncoder()
le_furnishing = LabelEncoder()

data["location"] = le_location.fit_transform(data["location"])
data["furnishing"] = le_furnishing.fit_transform(data["furnishing"])

# Features and target
X = data[["location", "size_sqft", "bhk", "furnishing"]]
y = data["rent"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "rent_model.pkl")
joblib.dump(le_location, "location_encoder.pkl")
joblib.dump(le_furnishing, "furnishing_encoder.pkl")

print("Model trained and saved successfully!")
