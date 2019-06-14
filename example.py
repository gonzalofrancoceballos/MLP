import models
import numpy as np
import pandas as pd

"""
Several reproducible examples for MLP model
"""


# Dummy function to create synthetic dataset
def f(X):
    """
    Simple function
    f(x0,x1,x2) = x0 + 2*x1 - x2**2
    
    :param X: input matrix with columns x0, x1, x2 (type: np.array)
    :return: f(X) (type: np.array) 
    
    """
    res = X[:,0] + 2*X[:, 1] - X[:,2]**2
    res = res.reshape([-1,1])
    
    return res


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
mlp = models.MLP(X, hidden_layers=[5, 4, 2], activation="tanh", optimizer="adam")

# Model train
mlp.train(X, y,
          X_dev=X_dev, 
          y_dev=y_dev,
          n_epoch=100,
          n_stopping_rounds=30)

# Run predict on new data
predictions = mlp.predict(X_pred)

# Evaluate model performance using the same metric it used to train
performance = mlp._compute_loss(predictions, y_pred)
print(f"Prediction loss: {performance}")


# CLASSIFICATION

# Get data
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
feature_cols = ["v1", "v2", "v3", "v4"]
iris.columns = feature_cols + ["class"]
iris["target"] = np.where(iris["class"] == "Iris-setosa", 1, 0)
X = iris[feature_cols].values
y = iris["target"].values.reshape([-1, 1])


# Instantiating model object
mlp = models.MLP(X, 
                 hidden_layers=[3, 3, 2],
                 activation="swish", 
                 optimizer="adam", 
                 problem="binary_classification",
                 loss="logloss")

# Model train (not usin dev this time)
mlp.train(X, y, n_epoch=1000, learning_rate=0.01)


# QUANTILE

# Creating synthetic dataset
N = 100000
X = 100 * (np.random.rand(N, 3) - 0.5)
X_dev = np.random.rand(N, 3)
X_pred = np.random.rand(N, 3)

noise = (np.random.normal(size=N)-0.5)/5
noise = noise.reshape([-1, 1])

# We need noise in the data for the quantile regression
y = f(X) * (1+noise)
y_dev = f(X_dev)
y_pred = f(X_pred)

# Instantiating model object for quantile 1
mlp_q1 = models.MLP(X,
                    hidden_layers=[5, 5, 5],
                    activation="tanh", optimizer="adam", 
                    problem="quantile",
                    loss="quantile",
                    q=0.01)

# Model train
mlp_q1.train(X, y,
             X_dev=X_dev, 
             y_dev=y_dev,
             n_epoch=100,
             n_stopping_rounds=30,
             verbose=False)


# Run predict on new data
predictions_q1 = mlp_q1.predict(X_pred)
print(f"Prediction average for quantile 1: {predictions_q1.mean()}")

# Instantiating model object for quantile 99
mlp_q99 = models.MLP(X, hidden_layers=[5, 5, 5],
                     activation="tanh", optimizer="adam",
                     problem="quantile",
                     loss="quantile",
                     q=0.99)

# Model train
mlp_q99.train(X, y,
              X_dev=X_dev, 
              y_dev=y_dev,
              n_epoch=100,
              n_stopping_rounds=30, 
              verbose=False)

# Run predict on new data
predictions_q99 = mlp_q99.predict(X_pred)
print(f"Prediction average for quantile 1: {predictions_q99.mean()}")
