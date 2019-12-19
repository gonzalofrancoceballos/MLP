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

from MLP.initializers import Glorot
from MLP.helpers import assign_activation
from abc import abstractmethod


class Layer:
    """
    Base class for layer
    """

    @abstractmethod
    def reset_layer(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def _from_dict(self, layer_dict: dict):
        pass

    @abstractmethod
    def update_delta(self, next_layer):
        pass

    @abstractmethod
    def update_gradients(self, a_in: np.array, reg_lambda: float):
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

    def __init__(
        self,
        units=None,
        activation=None,
        input_dim=None,
        kernel_initializer=Glorot(),
        initialize=False,
        layer_dict=None,
    ):

        """
        Initialize layer

        :param units: output size of the layer (type: int)
        :param input_dim: input dimension of the layer. If none, they will be inferred
        by the modelfrom the previous layer (type: int)
        :param activation: activation function of the layer (type: str)
        :param initialize: flag to initialize weights (type: bool)
        """

        self.W = None
        self.b = None
        self.delta = None
        self.db = None
        self.dW = None
        self.Z = None
        self.A = None

        if layer_dict is not None:
            self._from_dict(layer_dict)
        elif units is None or activation is None:
            raise AttributeError("It is necessary to especify units and activation")
        else:
            self.type = "dense"
            self._activation_type = activation.type
            self.activation = activation
            self.input_dim = input_dim
            self.output_dim = units
            self.initializer = kernel_initializer

        if initialize:
            self.reset_layer()

    def __repr__(self):
        return "[{}|{}] shape: {}\n".format(
            self.type, self.activation.type, self._get_shape()
        )

    def __str__(self):
        return "[{}|{}] shape: {}\n".format(
            self.type, self.activation.type, self._get_shape()
        )

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

    def update_delta(self, next_layer):
        """
        Computes and updates delta in back-propagation

        :param next_layer: next layer (type: Layer)
        """

        delta = np.matmul(next_layer.delta, next_layer.W.T) * self.activation.derivate(
            self.Z
        )
        self.delta = delta

    def update_gradients(self, a_in, reg_lambda):
        """
        Computes and updates gradients in back-propagation

        :param a_in: input matrix to the layer (type: np.array)
        :param reg_lambda: regularization factor (type: float)
        """

        delta_out = self.delta
        self.db = delta_out.sum(axis=0).reshape([1, -1])
        self.dW = np.matmul(a_in.T, delta_out)
        self.dW += reg_lambda * self.W

    def to_dict(self):
        layer_dict = {
            "type": self.type,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "activation": self.activation.type,
        }

        return layer_dict

    def _from_dict(self, layer_dict):
        """
        Populates weights from dict

        :param layer_dict:
        :return:
        """

        self.type = layer_dict["type"]
        self.W = np.array(layer_dict["W"])
        self.b = np.array(layer_dict["b"])
        self.activation = assign_activation(layer_dict["activation"])

    def _get_shape(self):
        return [self.input_dim, self.output_dim]
