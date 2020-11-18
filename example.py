from src import models
import numpy as np
import pandas as pd
from src.layers import Dense
from src.activations import Sigmoid, Relu, Linear
from src.losses import Logloss, MSE, Quantile


"""
Several reproducible examples for MLP model
"""


# Dummy function to create synthetic dataset
def f(x):
    """
    Simple function
    f(x0,x1,x2) = x0 + 2*x1 - x2**2

    :param x: input matrix with columns x0, x1, x2 (type: np.array)
    :return: f(X) (type: np.array)

    """
    res = x[:, 0] + 2 * x[:, 1] - x[:, 2] ** 2
    res = res.reshape([-1, 1])

    return res


print("---------------------------------------------")
print("---- Regression model -----------------------")
print("---------------------------------------------")
# REGRESSION
# Creating synthetic dataset
N = 100000
N_dev = 1000
X = np.random.rand(N, 3)
X_dev = np.random.rand(N, 3)
X_pred = np.random.rand(N, 3)
y = f(X)
y_dev = f(X_dev)
y_pred = f(X_pred)


# Instantiating model object
model = models.BasicMLP()
model.add(Dense(units=8, activation=Relu(), input_dim=X.shape[1]))
model.add(Dense(units=4, activation=Relu()))
model.add(Dense(units=4, activation=Relu()))
model.add(Dense(units=1, activation=Linear()))

# Model train
params = {"learning_rate": 0.001, "n_epoch": 100, "print_rate": 10}
loss = MSE()
print(model.layers)
model.train(loss, train_data=[X, y], params=params)

# Run predict on new data
predictions = model.predict(X_pred)

# Evaluate model performance using the same metric it used to train
performance = loss.forward(predictions, y_pred)
print(f"Prediction loss: {performance.mean()}")


print("---------------------------------------------")
print("---- Classification model -------------------")
print("---------------------------------------------")

print("Pulling Iris data from url...")
iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(iris_url, header=None)
feature_cols = ["v1", "v2", "v3", "v4"]
iris.columns = feature_cols + ["class"]
iris["target"] = np.where(iris["class"] == "Iris-setosa", 1, 0)
X = iris[feature_cols].values
y = iris["target"].values.reshape([-1, 1])


# Instantiating model object
print("Creating model...")
model = models.BasicMLP()
model.add(Dense(units=32, activation=Relu(), input_dim=X.shape[1]))
model.add(Dense(units=64, activation=Relu()))
model.add(Dense(units=8, activation=Relu()))
model.add(Dense(units=1, activation=Sigmoid()))

# Model train (not usin dev this time)
params = {"learning_rate": 0.001, "n_epoch": 100, "print_rate": 10}
loss = Logloss()
print("Starting train...")
model.train(loss, train_data=[X, y], params=params)


print("---------------------------------------------")
print("---- Quantile model -------------------------")
print("---------------------------------------------")
# Creating synthetic dataset
N = 100000
X = 100 * (np.random.rand(N, 3) - 0.5)
X_dev = np.random.rand(N, 3)
X_pred = np.random.rand(N, 3)

noise = (np.random.normal(size=N) - 0.5) / 5
noise = noise.reshape([-1, 1])

# We need noise in the data for the quantile regression
y = f(X) * (1 + noise)

# Instantiating model object for quantile 1
model = models.BasicMLP()
model.add(Dense(units=16, activation=Relu(), input_dim=X.shape[1]))
model.add(Dense(units=8, activation=Relu()))
model.add(Dense(units=4, activation=Relu()))
model.add(Dense(units=1, activation=Linear()))

# Model train
params = {"learning_rate": 0.001, "n_epoch": 100, "print_rate": 10}
loss = Quantile(0.01)
model.train(loss, train_data=[X, y], params=params)


# Run predict on new data
predictions_q1 = model.predict(X_pred)
print(f"Prediction average for quantile 1: {predictions_q1.mean()}")

# Instantiating model object for quantile 99
loss = Quantile(0.99)
model.train(loss, train_data=[X, y], params=params)

# Run predict on new data
predictions_q99 = model.predict(X_pred)
print(f"Prediction average for quantile 99: {predictions_q99.mean()}")
