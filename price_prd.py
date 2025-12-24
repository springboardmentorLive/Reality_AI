import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data\house_rent_prediction_2010_2024.csv')

x = data[['area' , 'romms']]
y = data['price']

model = LinearRegression()
model.fit(x , y)

new_house = [[1600 , 4]]
prediction_price = model.predict(new_house)

print("Predicted property price : $" , int(prediction_price[0]))