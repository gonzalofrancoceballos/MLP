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
from base import Loss


class MSE(Loss):
    """
    Class that implements Mean Squared Error
    """

    def __init__(self):
        self.type = "mse"

    def forward(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute MSE error between target and prediction
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise MSE 
        """
        return 0.5 * ((prediction - actual) ** 2)

    def derivate(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute the derivative of MSE error 
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of MSE 
        """
        return prediction - actual


class MAE(Loss):
    """
    Class that implements Mean Absolute Error
    """

    def __init__(self):
        self.type = "mae"

    def forward(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute MAE error between target and prediction
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise MAE 
        """
        return np.abs(prediction - actual)

    def derivate(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute the derivative of MAE 
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of MAE 
        """
        return np.where(prediction - actual > 0, 1, -1)


class Logloss(Loss):
    """
    Class that implements Logloss Error
    """

    def __init__(self):
        """
        Initialize logloss object
        eps is a small number to avoid extreme values in predictions of 0 and 1
        """
        self.type = "logloss"
        self._eps = 1e-15

    def forward(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute Logloss error between targt and prediction
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise Logloss
        """

        # Clip prediction to avioid 0 and 1
        prediction = np.clip(prediction, self._eps, 1 - self._eps)

        return -(actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction))

    def derivate(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute the derivative of Logloss error 
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of Logloss
        """

        # Clip prediction to avioid 0 and 1
        prediction = np.clip(prediction, self._eps, 1 - self._eps)
        return -(actual / prediction) + ((1 - actual) / (1 - prediction))


class Quantile(Loss):
    """
    Class that implements Quantile Loss
    """

    def __init__(self, q: float = 0.5):
        """
        Initialize quantile loss object
        :param q: quantile for which we want to cumpute the loss (type: float)
        """
        self.type = "quantile"
        self.q = q

    def forward(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute quantile loss for an especific quantile
        
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of Logloss
        """

        e = actual - prediction

        return np.maximum(self.q * e, (self.q - 1) * e)

    def derivate(self, actual: np.array, prediction: np.array) -> np.array:
        """
        Compute the derivative of quantile loss
        :param actual: target vector (type: np.array)
        :param prediction: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of Logloss
        """

        e = actual - prediction
        q_loss = np.where(e > 0, -self.q, 1 - self.q)
        return q_loss
