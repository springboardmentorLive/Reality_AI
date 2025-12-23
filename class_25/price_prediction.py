import pandas as pd
# 1. Fixed import syntax (dots instead of commas, lowercase 'from')
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/real_estate_prices.csv")

X = data[['area', 'rooms']]
y = data['price']

# 2. Fixed missing '=' assignment operator
model = LinearRegression()

model.fit(X, y)

new_house = [[1600, 4]]

# 3. Fixed missing '=' assignment operator
predicted_price = model.predict(new_house)

print("Predicted Property Price: ", int(predicted_price[0]))