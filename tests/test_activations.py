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

import unittest
import numpy as np
from MLP import activations
from numpy.testing import assert_array_almost_equal


class TestActivations(unittest.TestCase):
    def test_sigmoid_forward(self):
        activation = activations.Sigmoid()
        assert_array_almost_equal(np.array([0.5]), activation.forward(np.array([0])))

    def test_sigmoid_backward(self):
        activation = activations.Sigmoid()
        assert_array_almost_equal(
            np.array([0, 0]), activation.derivate(np.array([1000, -500]))
        )

    def test_relu_forward(self):
        activation = activations.Relu()
        assert_array_almost_equal(
            np.array([0, 1, 10]), activation.forward(np.array([-1, 1, 10]))
        )

    def test_relu_backward(self):
        activation = activations.Relu()
        assert_array_almost_equal(
            np.array([0, 1, 1]), activation.derivate(np.array([-1, 1, 10]))
        )

    def test_linear_forward(self):
        activation = activations.Linear()
        assert_array_almost_equal(
            np.array([-1, 1, 10]), activation.forward(np.array([-1, 1, 10]))
        )

    def test_linear_backward(self):
        activation = activations.Linear()
        assert_array_almost_equal(
            np.array([1, 1, 1]), activation.derivate(np.array([-1, 1, 10]))
        )

    def test_swish_forward(self):
        activation = activations.Swish()
        assert_array_almost_equal(
            np.array([0, 1000, 0]), activation.forward(np.array([0, 1000, -100]))
        )
        self.assertGreater(-0.1, activation.forward(-0.5))

    def test_swish_backward(self):
        activation = activations.Swish()
        assert_array_almost_equal(
            np.array([1, 0, 0.5]), activation.derivate(np.array([100, -100, 0]))
        )

    def test_leaky_relu_forward(self):
        for m in [0.01, 0.1, 1]:
            activation = activations.LeakyRelu(m=m)
            assert_array_almost_equal(
                np.array([-10 * m, -m, 1, 10]),
                activation.forward(np.array([-10, -1, 1, 10])),
            )

    def test_leaky_relu_backward(self):
        for m in [0.01, 0.1, 1]:
            activation = activations.LeakyRelu(m=m)
            assert_array_almost_equal(
                np.array([m, m, 1, 1]), activation.derivate(np.array([-10, -1, 1, 10]))
            )
