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
from initializers import Glorot
from abc import abstractmethod


class Layer:
    """
    Base class for layer
    """
    @abstractmethod
    def __init__(self):
        self.activation = None
        self.input_dim = None
        self.output_dim = None

    @abstractmethod
    def reset_layer(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass


class Dense(Layer):
    """
    Class that implements a dense layer
    Z = vector_product(X,W) + b
    A = activation(Z)
    where:
    X: input matrix of shape [m,input_dim]
    W: weights matrix of shape [input_dim, output_dim]
    b: bias vector of size [1, output_dim]
    """

    def __init__(self,
                 units,
                 activation,
                 input_dim=None,
                 kernel_initializer=Glorot(),
                 initialize=False):

        """
        Initialize layer
        When passed to

        :param units: output size of the layer (type: int)
        :param input_dim: input dimension of the layer. If none, they will be inferred
        by the modelfrom the previous layer (type: int)
        :param activation: activation function of the layer (type: str)
        :param initialize: flag to initialize weights (type: bool)
        """
        
        self._activation_name = activation.name
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = units
        self.initializer = kernel_initializer

        if initialize:
            self.reset_layer()
        
    def reset_layer(self):
        """
        Reset weights, bias and gradients of the layer
        """
        W, b, dW, db = self.initializer.initialize(self)
        self.W = W
        self.b = b
        self.dW = dW
        self.db = db
        
    def forward(self, x: np.array, update: bool = True) -> np.array:
        """
        Forward pass through layer

        :param x: input matrix to the layer (type: np.array)
        :param update: flag to update outputs Z and A. These values need to be 
        cached during train to compute the back-propagation pass

        :return: result of forward operation (type: np.array)
        """
        Z = np.matmul(x, self.W) + self.b
        A = self.activation.forward(Z)
        if update:
            self.Z = Z
            self.A = A
        return A