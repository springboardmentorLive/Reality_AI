import pandas as pd
import sklearn.linear_model import LinearRegression

data = pd.read_csv('data\house_rent_prediction_2010_2024.csv')

x=data[['year']]
y=data['price']

model=LinearRegression()
model.fit(x,y)

futur_years = [[2025], [2026], [2027]]
future_ptrices = model.predict(futur_years)

for year, price in zip(futur_years, future_ptrices):
    print(f"Year: {year[0]}, Predicted Price: {price}")