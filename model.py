# Daniel Cázares A01197517

import numpy as np

# Train - Test data split function 
def data_split(x, y, train_size = 0.8):
    n = int(train_size*len(x))
    
    x_train, y_train = x[:n], y[:n]
    x_test, y_test = x[n:], y[n:]
    
    return x_train, y_train, x_test, y_test

# Mean squared error function
def mse(y_real, y_pred):
    return np.mean((y_real - y_pred)**2)

# Root Mean squared error function
def rmse(y_real, y_pred):
    return np.sqrt(mse(y_real, y_pred))


# Linear regression classs. Has constructor, fit and predict functions
class LinearRegression:
    
    # Constructor
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        self.lr = learning_rate 
        self.iters = n_iterations # Number of iterations
        self.intercept = 0
        self.weights = None
    
    # Function to calculate the linear regression coefficients
    def fit(self, x, y):
        # Number of samples and features
        ns, nf = x.shape
        
        # Initial values for intercept and weights
        self.intercept = 0
        self.weights = np.zeros(nf)
        
        # Gradient Descent
        for i in range(self.iters):
            # Predict y values (Current Line equation)
            y_pred = self.intercept + np.dot(x, self.weights)
            
            # Error
            error = y_pred - y
            
            # Calculate gradients (partial derivatives) of coefficients
            grad_int = (2/ns) * np.sum(error)
            grad_ws = (2/ns) * np.dot(x.T, error)
            
            # Update coefficients
            self.intercept -= self.lr * grad_int
            self.weights -= self.lr * grad_ws

    # Function to predict values of y using the obtained intercept and weights in the fit function 
    def predict(self, x):
        return self.intercept + np.dot(x, self.weights)
