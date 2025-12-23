import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("data/real_estate_prices.csv")

# Define features (X) and target (y)
X = data[['year']]
y = data['price']

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict prices for future years
# Ensure inputs are formatted as a 2D array (list of lists)
future_years = [[2025], [2026], [2027]]
future_prices = model.predict(future_years)

# Print formatted results
for year, price in zip(future_years, future_prices):
    # Fixed f-string syntax using curly braces {}
    print(f"Year {year[0]} Predicted Price: {int(price)}")