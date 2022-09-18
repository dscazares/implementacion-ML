# Daniel Cázares A01197517

import pandas as pd
from model import LinearRegression, data_split, rmse, mse

# Read CSV. Taipei Real Estate Data from the Kaggle dataset 'Real estate price prediction' 
df = pd.read_csv('Real estate.csv', index_col=0)

# Labels (Y) and features (x) 
x = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']

# Train test data split
x_train, y_train, x_test, y_test = data_split(x, y)

# Features Scaling
x_train = (x_train-x_train.mean())/x_train.std()
x_test = (x_test-x_test.mean())/x_test.std()

# Model fit and prediction
model = LinearRegression()
model.fit(x_train, y_train)
preds = model.predict(x_test)

print("Equation: y = ", model.intercept, end ="")
for i in range(len(model.weights)):
    print(" +", model.weights[i], "x" + str(i+1), end ="")
print("\nMSE: ", mse(y_test, preds))
print("RMSE: ", rmse(y_test, preds))