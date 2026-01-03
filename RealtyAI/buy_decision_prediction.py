import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def predict_buy_decision(price, area, year):
    data = pd.read_csv("data/real_estate_prices.csv")

    def buy_decision(price, area, year):
        if price <= 6000000 and area >= 1300 and year >= 2020:
            return "Buy"
        else:
            return "Do Not Buy"

    data["decision"] = data.apply(
        lambda row: buy_decision(row["price"], row["area"], row["year"]),
        axis=1
    )

    X = data[["price", "area", "year"]]
    y = data["decision"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    prediction = model.predict([[price, area, year]])

    return prediction[0]