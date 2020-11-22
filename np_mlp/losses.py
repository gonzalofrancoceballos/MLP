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
from .tensor import Tensor


class Loss:
    """Base class for loss function for prediction error."""

    @property
    @abstractmethod
    def loss_type(self):
        pass

    @abstractmethod
    def forward(self, actual: Tensor, prediction: Tensor) -> Tensor:
        pass

    @abstractmethod
    def derivate(self, actual: Tensor, prediction: Tensor) -> Tensor:
        pass


class MSE(Loss):
    """
    Class that implements Mean Squared Error
    """

    @property
    def loss_type(self):
        return "mse"

    def forward(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute MSE error between target and prediction

        Args:
            actual: target vector
            prediction: predictions vector

        Returns:
            vector containing element-wise MSE

        """

        return 0.5 * ((prediction - actual) ** 2)

    def derivate(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute the derivative of MSE error.

        Args:
            actual: target vector
            prediction: vector containing element-wise derivative of MSE

        Returns:
           vector containing element-wise derivative of MSE

        """

        return prediction - actual


class MAE(Loss):
    """Class that implements Mean Absolute Error."""

    @property
    def loss_type(self):
        return "mae"

    def forward(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute MAE error between target and prediction.


        Args:
            actual: target vector
            prediction:  predictions vector

        Returns:
            vector containing element-wise MAE

        """

        return np.abs(prediction - actual)

    def derivate(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute the derivative of MAE.

        Args:
            actual: target vector
            prediction: vector containing element-wise derivative of MSE

        Returns:
           vector containing element-wise derivative of MAE

        """

        return np.where(prediction - actual > 0, 1, -1)


class Logloss(Loss):
    """Class that implements Logloss Error."""

    def __init__(self):
        """Initialize logloss object.

        `eps` is a small number to avoid extreme values in predictions of 0 and 1
        """

        self._eps = 1e-15

    @property
    def loss_type(self):
        return "logloss"

    def forward(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute Logloss error between targt and prediction.

        Args:
            actual: target vector
            prediction:  predictions vector

        Returns:
            vector containing element-wise Logloss

        """

        # Clip prediction to avioid 0 and 1
        prediction = np.clip(prediction, self._eps, 1 - self._eps)

        return -(actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction))

    def derivate(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute the derivative of Logloss error.

        Args:
            actual: target vector
            prediction: vector containing element-wise derivative of logloss

        Returns:
           vector containing element-wise derivative of logloss

        """

        # Clip prediction to avioid 0 and 1
        prediction = np.clip(prediction, self._eps, 1 - self._eps)
        return -(actual / prediction) + ((1 - actual) / (1 - prediction))


class Quantile(Loss):
    """Class that implements Quantile Loss."""

    def __init__(self, q: float = 0.5):
        """Initialize quantile loss object.

        Args:
            q: quantile for which we want to cumpute the loss
        """

        self.q = q

    @property
    def loss_type(self):
        return "quantile"

    def forward(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute quantile loss for an especific quantile

        Args:
            actual: target vector
            prediction:  predictions vector

        Returns:
            vector containing element-wise quantile loss

        """

        e = actual - prediction

        return np.maximum(self.q * e, (self.q - 1) * e)

    def derivate(self, actual: Tensor, prediction: Tensor) -> Tensor:
        """Compute the derivative of quantile loss.

        Args:
            actual: target vector
            prediction: vector containing element-wise derivative of logloss

        Returns:
           vector containing element-wise derivative of logloss

        """

        e = actual - prediction
        q_loss = np.where(e > 0, -self.q, 1 - self.q)
        return q_loss
