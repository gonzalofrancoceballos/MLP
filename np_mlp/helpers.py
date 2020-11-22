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

from . import activations, losses, optimizers


def assign_activation(activation_type: str) -> activations.Activation:
    """Returns actiavtion function given type.

    Args:
        activation_type: type of activation function

    Returns:
        activation function

    """

    options = {
        "sigmoid": activations.Sigmoid(),
        "swish": activations.Swish(),
        "relu": activations.Relu(),
        "leaky_relu": activations.LeakyRelu(),
        "tanh": activations.Tanh(),
        "linear": activations.Linear(),
    }

    return options[activation_type]


def assign_loss(loss_type: str) -> losses.Loss:
    """Returns loss function given type.

    Args:
        loss_type: type of loss funciton

    Returns:
        loss function

    """

    options = {
        "mse": losses.MSE(),
        "mae": losses.MAE(),
        "logloss": losses.Logloss(),
        "quantile": losses.Quantile(),
    }

    return options[loss_type]


def assign_optimizer(optimizer_type: str) -> optimizers.Optimizer:
    """Returns optimizer given type.

    Args:
        optimizer_type: type of optimizer

    Returns:
        optimizer object

    """

    options = {
        "gradient_descent": optimizers.GradientDescent(),
        "adam": optimizers.Adam(),
    }

    return options[optimizer_type]
