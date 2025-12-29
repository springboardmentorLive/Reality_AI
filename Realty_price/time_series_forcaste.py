import pandas as pd
from sklearn.linear_model import LinearRegression

# Data load karein
data = pd.read_csv("data/real_estate_prices.csv")

# Input (Year) aur Output (Price) define karein
X = data[['year']]
y = data['price']

# Model ko train karein
model = LinearRegression()
model.fit(X, y)

# Future years ke liye prediction karein
future_years = [[2025], [2026], [2027]]
future_prices = model.predict(future_years)

# Output dikhayein
for year, price in zip(future_years, future_prices):
    print(f"Year {year[0]} Predicted Price: ₹{int(price)}")