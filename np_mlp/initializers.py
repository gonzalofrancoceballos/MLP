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


class Initializer:
    @abstractmethod
    def initialize(self, layer) -> tuple:
        pass


class Rand(Initializer):
    """Implements a random weight initializer."""

    def initialize(self, layer) -> tuple:
        W = np.random.rand(layer.input_dim, layer.output_dim)
        b = np.random.rand(1, layer.output_dim)
        dW = np.zeros([layer.input_dim, layer.output_dim])
        db = np.zeros([1, layer.output_dim])
        return W, b, dW, db


class Glorot(Initializer):
    """Implements Gorot weight initializer."""

    def initialize(self, layer) -> tuple:
        if layer.activation.activation_type == "relu":
            W = np.random.normal(0, size=[layer.input_dim, layer.output_dim]) * np.sqrt(
                2 / layer.input_dim
            )
        else:
            W = np.random.normal(0, size=[layer.input_dim, layer.output_dim]) * np.sqrt(
                1 / layer.input_dim
            )

        b = np.zeros([1, layer.output_dim])
        dW = np.zeros([layer.input_dim, layer.output_dim])
        db = np.zeros([1, layer.output_dim])

        return W, b, dW, db
