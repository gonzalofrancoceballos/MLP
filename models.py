"""
This file is part of MLP project <github.com/gonzalofrancoceballos/MLP>
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
- Mini-batch (testing)
- Train-dev with early stopping (testing)
- Validate classification
- Quantile regression
- Load/save model functionality
"""

import numpy as np
import activations
import losses
import optimizers
from data_processing import Batcher
from layers import Dense


# Model
class MLP():
    """
    Multi-layer perceptron
    """
    def __init__(self, 
                 X, 
                 hidden_layers = [2,2], 
                 activation = "relu",
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
            self._loss = losses.MSE()
        if loss == "logloss":
            self._loss = losses.Logloss()
        
        # Output activation
        if problem == "regression" : 
            self._layers[-1].activation = activations.Linear()
        if problem == "quantile" : 
            self._layers[-1].activation = activations.Linear()
        if problem == "binary_classification":
            self._layers[-1].activation = activations.Sigmoid()
        
        # Set optimizer
        if optimizer == "gradient_descent":
            self._optimizer = optimizers.Gradient_descent()
        if optimizer == "adam":
            self._optimizer = optimizers.Adam()
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
            

    def train(self, X, y, X_dev=None, y_dev=None,
              n_epoch=100, batch_size=128, n_stopping_rounds=10, 
              learning_rate=0.0001, reg_lambda=0.01, verbose=True):
        """
        Run several train steps
        :param X: input matrix to the network (type: np.array)
        :param y: target vector (type: np.array)
        :param X_dev: development matrix for early stopping (type: np.array)
        :param y_dev: development target vector (type: np.array)
        :param n_iter: number of train iterations (type: int)
        :param n_epoch: number of epochs (type: int)
        :param batch_size: batch size (type: int)
        :param n_stopping_rounds: number of iterations before early stopping (type: int)
        :param learning_rate: learning rate fot the optimizer (type: float)
        :param reg_lambda: weights regulariazation factor (type: float)
        :param verbose: to plot train loss (type bool)
        """
        
        self._optimizer.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self._batch_size = batch_size
        self._n_epoch = n_epoch
        self._batcher = Batcher([X,y], batch_size=self._batch_size)
        self._n_stopping_rounds = n_stopping_rounds
        early_stopping = False
        epoch = 1
        best_loss = 1e14
        early_stopping_counter = 0
        
        # Start train
        while epoch <=self._n_epoch and early_stopping_counter<self._n_stopping_rounds:
            train_loss = []
            self._batcher.reset()
            for batch_i in range(self._batcher.n_batches):
                X_batch, y_batch = self._batcher.next()
                self._train_step(X_batch, y_batch)
                train_loss.append(self._compute_loss(self._layers[-1].A,y_batch))
                
            if type(X_dev) == np.ndarray and type(y_dev) == np.ndarray:
                dev_pred = self.predict(X_dev)
                dev_loss = self._compute_loss(dev_pred,y_dev)
                
                if best_loss > dev_loss:
                    early_stopping_counter = 0
                    best_loss = dev_loss
                else:
                    early_stopping_counter += 1
                    
                if verbose:
                    print(f"epoch: {epoch} | train_loss: {np.mean(train_loss)} |  dev_loss: {dev_loss}") 
                        
            else:
                if verbose:
                    print(f"epoch: {epoch} | train_loss: {np.mean(train_loss)}")
                
            epoch = epoch+1

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
