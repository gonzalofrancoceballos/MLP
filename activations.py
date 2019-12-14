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

from base import Activation
from model_utils import sigmoid
import numpy as np


class Sigmoid(Activation):
    """
    Sigmoid activation function
    """

    def __init__(self):
        """
        Initialize object
        """
        self.type = "sigmoid"

    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return (1 - self.forward(x)) * self.forward(x)


class Swish(Activation):
    """
    Swish activation function
    """

    def __init__(self):
        """
        Initialize object
        """
        self.type = "swish"

    @staticmethod
    def _sigmoid_p(x):
        return (1 - sigmoid(x)) * sigmoid(x)

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
        return x * self._sigmoid_p(x) + sigmoid(x)


class Relu(Activation):
    """
    ReLu activation function
    """

    def __init__(self):
        """
        Initialize object
        """
        self.type = "relu"

    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x > 0, x, 0.0)

    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x > 0, 1.0, 0.0)


class LeakyRelu(Activation):
    """
    Leaky activation function
    """

    def __init__(self, m=0.01):
        """
        Initialization of the activation function
        :param m: slope of he function (type: float)
        """
        self.m = m
        self.type = "leaky_relu"

    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x > 0, x, self.m * x)

    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return np.where(x > 0, 1.0, self.m)


class Tanh(Activation):
    """
    Tanh activation function
    """

    def __init__(self):
        """
        Initialize object
        """
        self.type = "tanh"

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
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """
    Linear activation function
    """

    def __init__(self):
        """
        Initialize object
        """
        self.type = "linear"

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


class Softmax(Activation):
    """
    Softmax activation function
    """

    def __init__(self):
        """
        Initialize object
        """
        self.type = "softmax"

    def forward(self, x):
        """
        Forward propagation operation
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1).reshape([-1, 1])

    def derivate(self, x):
        """
        Derivative of the activation function at each point of the input tensor
        :param x: tensor apply operation element-wise (type: np.array)
        :return: result of the operation (type: np.array)
        """
        return (1 - self.forward(x)) * self.forward(x)
