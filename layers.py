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

import numpy as np
import activations


def glorot_initializer(input_dim, output_dim, activation):
    if activation == "relu":
        return np.random.normal(0, size=[input_dim, output_dim])*  np.sqrt(2/input_dim)
    else:
        return np.random.normal(0, size=[input_dim, output_dim])*  np.sqrt(1/input_dim)
        
    
    
    
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
        
        self._activation_str = activation
        
        if activation == "sigmoid": 
            self.activation = activations.Sigmoid()
        if activation == "relu": 
            self.activation = activations.Relu()
        if activation == "leaky_relu":
            self.activation = activations.Leaky_relu()
        if activation == "tanh": 
            self.activation = activations.Tanh()
        if activation == "linear": 
            self.activation = activations.Linear()
        if activation == "swish": 
            self.activation = activations.Swish()
            
        self.reset_layer(input_dim, output_dim)
        
    def reset_layer(self, input_dim, output_dim, glorot=True):
        """
        Reset weights, bias and gradients of the layer
        """
        if glorot:
            self.W = glorot_initializer(input_dim, output_dim, self._activation_str)
            self.b = np.zeros([1, output_dim])
        else:
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