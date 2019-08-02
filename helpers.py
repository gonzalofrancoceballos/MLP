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

import activations
import losses
import optimizers


def assign_activation(name: str) -> activations.Activation:
    """
    Returns actiavtion function given name

    :param name: name of activation funciton (type: str)

    :return: actiavtion function (type: Activation)
    """

    options = {"sigmoid": activations.Sigmoid(),
               "swish": activations.Swish(),
               "relu": activations.Relu(),
               "leaky_relu": activations.LeakyRelu(),
               "tanh": activations.Tanh(),
               "linear": activations.Linear()
               }

    return options[name]


def assign_loss(name: str) -> losses.Loss:
    """
    Returns loss function given name

    :param name: name of loss funciton (type: str)

    :return: loss function (type: Loss)
    """

    options = {"mse":  losses.MSE(),
               "mae": losses.MAE(),
               "logloss": losses.Logloss(),
               "quantile": losses.Quantile()
               }

    return options[name]


def assign_optimizer(name: str) -> optimizers.Optimizer:
    """
    Returns optimizer given name

    :param name: name of optimizer (type: str)

    :return: optimizer object (type: Optimizer )
    """

    options = {"gradient_descent": optimizers.GradientDescent(),
               "adam": optimizers.Adam()}

    return options[name]