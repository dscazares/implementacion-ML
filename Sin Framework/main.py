# Daniel Cázares A01197517

import pandas as pd
from model import LinearRegression, data_split, rmse

# Read CSV
df = pd.read_csv('Real estate.csv')

# Labels (Y) and features (x) 
x = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']

# Train test data split
x_train, y_train, x_test, y_test = data_split(x, y)

# Training Features Escalation
x_train=(x_train-x_train.mean())/x_train.std()

# Model fit and prediction
model = LinearRegression()
model.fit(x_train, y_train)
preds = model.predict(x_test)

print("Equation: y = ", model.intercept, end ="")
for w in model.weights:
    print(" +", w, "x", end ="")
print("\nRMSE: ", rmse(y_test, preds))