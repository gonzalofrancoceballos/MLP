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
from MLP import losses
from unittest import TestCase
from numpy.testing import assert_array_almost_equal

actual_regression = np.array([0, 1, 2, 3])
prediction_regression = np.array([1, 1, 1, -1])

actual_binary = np.array([0, 0, 1, 1])
prediction_binary = np.array([0.01, 0.2, 0.75, 0.99])


class TestLosses(TestCase):
    def test_mse_forward(self):
        loss = losses.MSE()
        result = loss.forward(actual_regression, prediction_regression)
        expected = 0.5 * np.array([1, 0, 1, 16])
        assert_array_almost_equal(expected, result)

    def test_mse_backward(self):
        loss = losses.MSE()
        result = loss.derivate(actual_regression, prediction_regression)
        expected = np.array([1, 0, -1, -4])
        assert_array_almost_equal(expected, result)

    def test_mae_forward(self):
        loss = losses.MAE()
        result = loss.forward(actual_regression, prediction_regression)
        expected = np.array([1, 0, 1, 4])
        assert_array_almost_equal(expected, result)

    def test_mae_backward(self):
        loss = losses.MAE()
        result = loss.derivate(actual_regression, prediction_regression)
        expected = np.array([1, -1, -1, -1])
        assert_array_almost_equal(expected, result)

    def test_logloss_forward(self):
        loss = losses.Logloss()
        result = loss.forward(actual_binary, prediction_binary)
        expected = np.array([0.01005034, 0.22314355, 0.28768207, 0.01005034])
        assert_array_almost_equal(expected, result)

    def test_logloss_backward(self):
        loss = losses.Logloss()
        result = loss.derivate(actual_binary, prediction_binary)
        expected = np.array([1.01010101, 1.25, -1.33333333, -1.01010101])
        assert_array_almost_equal(expected, result)

    def test_quantile_forward(self):
        loss = losses.Quantile(0.25)
        result = loss.forward(actual_regression, prediction_regression)
        expected = np.array([0.75, 0, 0.25, 1])
        assert_array_almost_equal(expected, result)

    def test_quantile_backward(self):
        loss = losses.Quantile(0.25)
        result = loss.derivate(actual_regression, prediction_regression)
        expected = np.array([0.75, 0.75, -0.25, -0.25])
        assert_array_almost_equal(expected, result)
