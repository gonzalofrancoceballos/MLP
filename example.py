import models
import numpy as np
import pandas as pd

"""
Several reproducible examples for MLP model
"""


## REGRESSION

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
mlp = models.MLP(X, hidden_layers=[5,4,2], activation="tanh", optimizer="adam")

# Model train
mlp.train(X,y,
          X_dev=X_dev, 
          y_dev=y_dev,
          n_epoch=100,
          n_stopping_rounds=30)

# Run predict on new data
predictions = mlp.predict(X_pred)

# Evaluate model performance using the same metric it used to train
performance = mlp._compute_loss(predictions,y_pred)
print(f"Prediction loss: {performance}")


## CLASSIFICATION

# Get data
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
feature_cols = ["v1", "v2", "v3","v4"]
iris.columns = feature_cols +  ["class"]
iris["target"] = np.where(iris["class"] == "Iris-setosa", 1, 0)
X = iris[feature_cols].values
y = iris["target"].values.reshape([-1,1])


# Instantiating model object
mlp = models.MLP(X, 
                 hidden_layers=[3,3,2], 
                 activation="swish", 
                 optimizer="adam", 
                 problem="binary_classification",
                 loss = "logloss")

# Model train (not usin dev this time)
mlp.train(X,y, n_epoch=1000,learning_rate=0.01)