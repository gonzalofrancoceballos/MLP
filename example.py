"""
Snippet of code run a simple MLP model
"""

from MLP import MLP
import numpy as np

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
y = f(X)
y_dev = f(X_dev)


# Instantiating model object
mlp = MLP(X, hidden_layers=[2,2,3], activation="relu", optimizer="adam")

# Running train
mlp.train(X,y, n_epoch=100, verbose=True)