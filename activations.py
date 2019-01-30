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
    def __init__(self):
        """
        Initialize object
        """
        self.name = "sigmoid"
        
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
    
    def __init__(self):
        """
        Initialize object
        """
        self.name = "swish"
        
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
    def __init__(self):
        """
        Initialize object
        """
        self.name = "relu"
        
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
        self.name = "leaky_relu"
        
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
    
    def __init__(self):
        """
        Initialize object
        """
        self.name = "tanh"
        
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
    
    def __init__(self):
        """
        Initialize object
        """
        self.name = "linear"
        
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