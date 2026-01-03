import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data/real_estate_prices.csv")

# Features and target
X = data[['area', 'rooms']]
y = data['price']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# New house data (area, rooms)
new_house = [[1600, 4]]

# Predict price
predicted_price = model.predict(new_house)

print("Predicted Property Price:", int(predicted_price[0]))