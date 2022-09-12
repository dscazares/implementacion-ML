# Daniel Cázares A01197517

import pandas as pd
from model import LinearRegression, data_split, rmse

df = pd.read_csv('Real estate.csv')

x = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']

x_train, y_train, x_test, y_test = data_split(x, y)

model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

error = rmse(y_test, y_pred)
print("Error", error)