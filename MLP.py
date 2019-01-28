"""
Simple and light-weight implementation of a Multi-Layer Perceptron using Numpy

Copyright 2019 Gonzalo Franco Ceballos <gonzalofrancoceballos@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
TODO:
- Mini-batch
- Train-dev with early stopping
- Validate classification
- Quantile regression
- Load/save model functionality
"""

import numpy as np


# Activation functions
def sigmoid(x):
    """
    Sigmoid function
    :param x: input matrix (type:np.array)
    :output: result of applying sigmoid functin element-wise
    """
    return 1. / (1. + np.exp(-x))


class Sigmoid():
    """
    Sigmoid activation function
    """
    
    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return sigmoid(x)
        
    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return (1-sigmoid(x)) * sigmoid(x)

class Swish():
    """
    Swish activation function
    """
   
    def _sigmoid_p(self, x):
        return (1-sigmoid(x)) * sigmoid(x)
    
    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return x * sigmoid(x)
        
    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return x*self._sigmoid_p(x) + sigmoid(x)

class Relu():
    """
    ReLu activation function
    """
    
    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x>0, x, 0.)
    
    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x>0, 1., 0.)

class Leaky_relu():
    """
    Leaky activation function
    """
    def __init__(self, m=0.01):
        """
        Initialization of the activation function
        :param m: slope of he function (type: float)
        """
        self.m = m
        
    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x>0, x, self.m*x)
    
    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x>0, 1., self.m)
    
class Tanh():
    """
    Tanh activation function
    """
    
    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.tanh(x)
    
    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return 1-np.tanh(x)**2
    
    
class Linear():
    """
    Linear activation function
    """
    
    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return x
    
    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return 1
    
    
# Losses
class MSE():
    """
    Class that implements Mean Squared Error
    """
    def forward(self, actual, prediction):
        """
        Compute MSE error between targt and prediction
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise MSE 
        """
        return 0.5*((prediction-actual)**2) 
    
    def derivate(self, actual, prediction):
        """
        Compute the derivative of MSE error 
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of MSE 
        """
        return prediction - actual
    
class Logloss():
    """
    Class that implements Logloss Error
    """
    def forward(self, actual, prediction):
        """
        Compute Logloss error between targt and prediction
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise Logloss
        """
        return actual*np.log(prediction) + (1-actual)*np.log(1-prediction)
    
    def derivate(self, actual, prediction):
        """
        Compute the derivative of Logloss error 
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of Logloss
        """
        return (actual-prediction)/(prediction*(1-actual))
    
    
# Layers
class Dense():
    """
    Class that implements a dense layer
    Z = vector_product(X,W) + b
    A = activation(Z)
    where:
    X: input matrix of shape [m,input_dim]
    W: weights matrix of shape [input_dim, output_dim]
    b: bias vector of size [1, output_dim]
    """
    def __init__(self, input_dim, output_dim, activation = "sigmoid"):
        """
        Initialize layer
        :param input_dim: input dimension of the layer (type: int)
        :param output_dim: output dimension of the layer (type: int)
        :param activation: activation function of the layer (type: str)
        """
    
        if activation == "sigmoid": 
            self.activation = Sigmoid()
        if activation == "relu": 
            self.activation = Relu()
        if activation == "leaky_relu":
            self.activation = Leaky_relu()
        if activation == "tanh": 
            self.activation = Tanh()
        if activation == "linear": 
            self.activation = Linear()
        if activation == "swish": 
            self.activation = Swish()
            
        self.reset_layer(input_dim, output_dim)
        
    def reset_layer(self, input_dim, output_dim):
        """
        Reset weights, bias and gradients of the layer
        """
        self.W = np.random.rand(input_dim, output_dim)
        self.b = np.random.rand(1, output_dim)
        self.dW = np.zeros([input_dim, output_dim])
        self.db = np.zeros([1, output_dim])
        
    def forward(self, X, update=True):
        """
        Forward pass through layer
        :param X: input matrix to the layer (type: np.array)
        :param update: flag to update outputs Z and A. These values need to be 
        cached during train to compute the back-propagation pass
        :return: result of forward operation (type: np.array)
        """
        Z = np.matmul(X, self.W) + self.b
        A = self.activation.forward(Z)
        if update:
            self.Z = Z
            self.A = A
        return A 
    
    
# Optimizers
class Gradient_descent():
    """
    Implements gradient descent optimizer
    """
    def __init__(self, learning_rate=0.001):
        """
        Initialize optimizer
        :param learning rate: learning rate of each iteration (type: float)
        """
        self.learning_rate = learning_rate
        
    def update_weights(self, layers):
        """
        Perform update rule
        :param layers: layers of the MLP to update (type: list[Dense()])
        :return: layers with updated weights (type: list[Dense()])
        """
        for i in range(len(layers)):
            layers[i].W = layers[i].W - self.learning_rate * layers[i].dW
            layers[i].b = layers[i].b - self.learning_rate * layers[i].db
        
        return layers
    
    
class Adam():
    """
    Implements Adam optimizer
    """
    def __init__(self, learning_rate=0.001):
        """
        Initialize optimizer
        :param learning rate: learning rate of each iteration (type: float)
        """
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.t = 1
        
    def initialize_parameters(self, layers):
        """
        Initializes momemtum and velocity parameters for each layer of MLP
        :param layers: layers of the MLP (type: list[Dense()])
        :return: layers with initialized parameters (type: list[Dense()])
        """
        for i, layer in enumerate(layers):
            adam = {
                "mW" : np.zeros(layer.dW.shape),
                "mb" : np.zeros(layer.db.shape),
                "vW" : np.zeros(layer.dW.shape),
                "vb" : np.zeros(layer.db.shape)}
            layers[i].adam = adam
        return layers

        
    def update_weights(self, layers):
        """
        Perform update rule
        :param layers: layers of the MLP to update (type: list[Dense()])
        :return: layers with updated weights (type: list[Dense()])
        """
        t = self.t
        for i, layer in enumerate(layers):            
            adam  = {
                "mW" : (self.beta_1*layer.adam["mW"] + (1-self.beta_1)*layer.dW),
                "mb" : (self.beta_1*layer.adam["mb"] + (1-self.beta_1)*layer.db),
                "vW" : (self.beta_2*layer.adam["vW"] + (1-self.beta_2)*layer.dW**2),
                "vb" : (self.beta_2*layer.adam["vb"] + (1-self.beta_2)*layer.db**2)}
            
            layer.adam = adam
            
            mW_corrected  = adam["mW"] / (1-(self.beta_1**t))
            mb_corrected  = adam["mb"] / (1-(self.beta_1**t))
            vW_corrected  = adam["vW"] / (1-(self.beta_2**t))
            vb_corrected  = adam["vb"] / (1-(self.beta_2**t))
            
            layer.W = layer.W - (self.learning_rate * mW_corrected/(np.sqrt(vW_corrected)+ self.epsilon))
            layer.b = layer.b - (self.learning_rate * mb_corrected/(np.sqrt(vb_corrected)+ self.epsilon))
            
            layers[i] = layer
        self.t = t+1
        return layers
    

# Model
class MLP():
    """
    Multi-layer perceptron
    """
    def __init__(self, 
                 X, 
                 hidden_layers = [2,2], 
                 activation = "sigmoid",
                 loss = "mse", 
                 problem="regression",
                 optimizer ="gradient_descent"):
        """
        Initialize network
        
        :param X: input data (type: np.array)
        :param hidden layers: size of hidden layers (type: list[int])
        :param activation: activartion (type: str)
        :param loss: loss function to be used(type: str)
        :param problem: regression, binary_classification, quantile (type: str)
        :param optimizer: optimizer to use (type: str)
        """
        
        self.print_rate = 100
        self._activation = activation
        self.dims = [X.shape[1]] + hidden_layers + [1]

        # Create layers
        self._build_architecture()
        
        # Losses
        if loss == "mse":
            self._loss = MSE()
        if loss == "logloss":
            self._loss = Logloss()
        
        # Output activation
        if problem == "regression" : 
            self._layers[-1].activation = Linear()
        if problem == "quantile" : 
            self._layers[-1].activation = Linear()
        if problem == "binary_classification":
            self._layers[-1].activation = Sigmoid()
        
        # Set optimizer
        if optimizer == "gradient_descent":
            self._optimizer = Gradient_descent()
        if optimizer == "adam":
            self._optimizer = Adam()
            self._layers = self._optimizer.initialize_parameters(self._layers)
            
            
    def _build_architecture(self):
        """
        Build architecture of MLP. 
        Instantiates Dense layers inside of a list
        """
        
        self._layers = []
        for input_dim, output_dim in  zip(self.dims[:-1], self.dims[1:]):
            self._layers.append(Dense(input_dim, output_dim, activation=self._activation))
        self.n_layers = len(self._layers)
        
    def _forward_prop(self, X, update=True):
        """
        Computes a forward pass though the architecture of the network
        :param X: input matrix to the network (type: np.array)
        :return: output of the network (type: np.array)
        """
        
        A = self._layers[0].forward(X, update=update)
        for layer in self._layers[1:]:
            A = layer.forward(A, update=update)
            
        return A
    
    def predict(self, X):
        """
        Computes a forward pass and returns prediction
        Note that this operation will not update Z and A of each weight
        :param X: input matrix to the network (type: np.array)
        :return: output of the network (type: np.array)
        """
        pred = self._forward_prop(X, update=False)
        return pred
            
    def _back_prop(self, X, y):
        """
        Computes back-propagation pass through the network
        It retrieves output of the final layer, self._layers[-1].A, and back-propagates
        its error through the layers of the network, computing and updating its gradients
        :param X: input matrix to the network (type: np.array)
        :param y: target vector (type: np.array)
        """
        for i in np.arange(len(self._layers))+1:
            # Compute deltas deltas 
            layer = self._layers[-i]
            if i==1:
                delta = self._loss.derivate(y, layer.A) * layer.activation.derivate(layer.Z)
            else:
                i_next = i-1
                layer_next = self._layers[-i_next]
                delta = np.matmul(layer_next.delta, layer_next.W.T) * layer.activation.derivate(layer.Z)
            self._layers[-i].delta = delta
            
            # Compute gradients
            if i==self.n_layers:
                a_in = X
            else:
                i_prev = i+1
                a_in = self._layers[-i_prev].A
            delta_out = self._layers[-i].delta
            self._layers[-i].db = delta_out.sum(axis=0).reshape([1,-1])
            self._layers[-i].dW = np.matmul(a_in.T,  delta_out) 
            self._layers[-i].dW += self.reg_lambda * self._layers[-i].W
            
    def _train_step(self, X, y):
        """
        Performs a complete train step of the network
        (1) Forward pass: computes Z,A for each layer
        (2) Back propagation: computes gradients for each layer
        (3) Update weights: call optimizer to perform update rule
        
        :param X: input matrix to the network (type: np.array)
        :param y: target vector (type: np.array)
        """
        # Forward propagation
        _ = self._forward_prop(X)

        # Back propagation
        self._back_prop(X,y)

        # Update rule
        self._layers = self._optimizer.update_weights(self._layers)
            

    def train(self, X, y, n_iter=100, learning_rate=0.0001, reg_lambda=0.01,  verbose=True):
        """
        Run several train steps
        :param X: input matrix to the network (type: np.array)
        :param y: target vector (type: np.array)
        :param n_iter: number of train iterations
        :param learning_rate: learning rate fot the optimizer
        :param reg_lambda: weights regulariazation factor
        :param verbose: to plot train loss
        """
        self._optimizer.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        for i in range(n_iter):
            self._train_step(X,y)         
            if  verbose and i%self.print_rate==0:
                train_loss = self._compute_loss(self._layers[-1].A,y)
                print(f"iter: {i}  loss: {train_loss}")
                

    def _compute_loss(self, actual, prediction):
        """
        Computes loss between prediction and target
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: average loss (type: float)
        """
        
        current_loss = self._loss.forward(actual, prediction)
        current_loss = np.mean(current_loss)
        return current_loss
