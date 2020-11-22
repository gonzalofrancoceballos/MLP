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
from abc import abstractmethod
from .model_utils import sigmoid


class Activation:
    """Base class for activation function of a layer."""

    @property
    @abstractmethod
    def activation_type(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivate(self, x):
        pass


class Sigmoid(Activation):
    """Sigmoid activation function."""

    @property
    def activation_type(self):
        return "sigmoid"

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation
        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:

        """
        return (1 - self.forward(x)) * self.forward(x)


class Swish(Activation):
    """Swish activation function."""

    @property
    def activation_type(self):
        return "swis"

    @staticmethod
    def _sigmoid_p(x):
        return (1 - sigmoid(x)) * sigmoid(x)

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return x * sigmoid(x)

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation
        """
        return x * self._sigmoid_p(x) + sigmoid(x)


class Relu(Activation):
    """ReLu activation function."""

    @property
    def activation_type(self):
        return "relu"

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation
        """
        return np.where(x > 0, x, 0.0)

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return np.where(x > 0, 1.0, 0.0)


class LeakyRelu(Activation):
    """Leaky Relu activation function"""

    def __init__(self, m=0.01):
        """Initialization of the activation function.

        Args:
            m: slope of the function
        """
        self.m = m

    @property
    def activation_type(self):
        return "leaky_relu"

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return np.where(x > 0, x, self.m * x)

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return np.where(x > 0, 1.0, self.m)


class Tanh(Activation):
    """Tanh activation function."""

    @property
    def activation_type(self):
        return "tanh"

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return np.tanh(x)

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return 1 - np.tanh(x) ** 2


class Linear(Activation):
    """Linear activation function."""

    @property
    def activation_type(self):
        return "linear"

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return x

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return 1


class Softmax(Activation):
    """Softmax activation function."""

    @property
    def activation_type(self):
        return "softmax"

    def forward(self, x):
        """Forward propagation operation.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1).reshape([-1, 1])

    def derivate(self, x):
        """Derivative of the activation function at each point of the input tensor.

        Args:
            x: tensor apply operation element-wise

        Returns:
            result of the operation

        """
        return (1 - self.forward(x)) * self.forward(x)
