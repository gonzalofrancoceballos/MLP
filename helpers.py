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

import base
import activations
import losses
import optimizers


def assign_activation(type: str) -> base.Activation:
    """
    Returns actiavtion function given type

    :param type: type of activation funciton (type: str)

    :return: actiavtion function (type: Activation)
    """

    options = {"sigmoid": activations.Sigmoid(),
               "swish": activations.Swish(),
               "relu": activations.Relu(),
               "leaky_relu": activations.LeakyRelu(),
               "tanh": activations.Tanh(),
               "linear": activations.Linear()
               }

    return options[type]


def assign_loss(type: str) -> base.Loss:
    """
    Returns loss function given type

    :param type: type of loss funciton (type: str)

    :return: loss function (type: Loss)
    """

    options = {"mse":  losses.MSE(),
               "mae": losses.MAE(),
               "logloss": losses.Logloss(),
               "quantile": losses.Quantile()
               }

    return options[type]


def assign_optimizer(type: str) -> base.Optimizer:
    """
    Returns optimizer given type

    :param type: type of optimizer (type: str)

    :return: optimizer object (type: Optimizer )
    """

    options = {"gradient_descent": optimizers.GradientDescent(),
               "adam": optimizers.Adam()}

    return options[type]
