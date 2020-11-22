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
from typing import List
from . import model_utils

from abc import abstractmethod
from .layers import Dense, Layer
from .losses import Loss
from .train import ModelTrain
from .optimizers import Optimizer, Adam

"""
TODO:
- Save train log
- Auto-save best train when using cross-validation
- Keep trainlog
- 2D Conv layer
- 3D Conv layer
- Flatten layer
- Concat layer
"""


class Model:
    """
    Base class for a model
    """

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def forward_prop(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def back_prop(self, x: np.array, y: np.array, loss: Loss, reg_lambda: float):
        pass

    @abstractmethod
    def add(self, layer: Layer):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def reset_layers(self):
        pass


class BasicMLP(Model):
    """Multi-layer perceptron."""

    def __init__(self, model_dict: dict = None):
        """Initialize network.

        Args:
            model_dict: python dictionary containing all necessary information to
            instantiate an existing model

        """

        self.reg_lambda = 0.01
        self.layers = []
        self.n_layers = 0
        self._model_dict = model_dict
        self._trainer = None
        self.train_log = None
        if self._model_dict is not None:
            self._build_architecture_from_dict()

    def __repr__(self):
        if self.layers:
            repr_str = "BasicMLP model:\n"
            for layer in self.layers:
                repr_str = repr_str + str(layer)

            return repr_str
        else:
            return "Empty BasicMLP model"

    def __str__(self):
        if self.layers:
            repr_str = ""
            for layer in self.layers:
                repr_str = repr_str + str(layer)

            return repr_str
        else:
            return "Empty BasicMLP model"

    def add(self, layer: Layer):
        """Adds a layer to the model.

        Args:
            layer: layer to be added

        Returns:

        """

        if len(self.layers) == 0:
            if layer.input_dim is None:
                raise AttributeError(
                    "It is necessary to especify input dim for first layer"
                )
        else:
            layer.input_dim = self.layers[-1].output_dim

        layer.reset_layer()
        self.layers.append(layer)
        self.n_layers += 1

    def compile(self):
        """Compiles model before train. For now, it just resets layers.
        For now, this is just a dummy method to reflect the same behavior as Keras.
        """

        self.reset_layers()

    def predict(self, x: np.array) -> np.array:
        """Computes a forward pass and returns prediction.

        Note that this operation will not update Z and A of each weight,
        this must only happen during train

        Args:
            x: input matrix to the network

        Returns:
            output of the network

        """

        pred = self.forward_prop(x, update=False)
        return pred

    def train(
        self,
        loss: Loss,
        train_data: list,
        optimizer: Optimizer = Adam(),
        dev_data: list = None,
        params: dict = None,
    ):
        """Perform train operation.

        Args:
            loss: loss function
            train_data: train data
            optimizer: optimizar to use
            dev_data: data to use for early-stopping
            params: parameters for train

        Returns:

        """

        self._trainer = ModelTrain()
        self._trainer.train(self, loss, train_data, optimizer, dev_data, params)

    def forward_prop(self, x: np.array, update: bool = True):
        """Computes one forward pass though the architecture of the network.

        Args:
            x: input matrix to the network
            update: flag to update latest values through the network

        Returns:
            output of the network

        """

        A = x
        for layer in self.layers:
            A = layer.forward(A, update=update)

        return A

    def back_prop(self, x: np.array, y: np.array, loss: Loss, reg_lambda: float = 0.01):
        """Computes back-propagation pass through the network.

        It retrieves output of the final layer, self.layers[-1].A,
        and back-propagates its error through the layers of the network,
        computing and updating its gradients

        Args:
            x: input matrix to the network
            y: target vector
            loss: loss funtion object
            reg_lambda: regularizatiopn factor for gradients

        """

        self._update_deltas(loss, y)
        self._update_gradients(x, reg_lambda)

    def _update_deltas(self, loss: Loss, y: np.array):
        """Starting from last layer, compute and update deltas in reverse order.

        Args:
            loss: loss function object
            y: target verctor

        """

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                delta = loss.derivate(y, layer.A) * layer.activation.derivate(layer.Z)
                layer.delta = delta
            else:
                layer_next = self.layers[-i]
                layer.update_delta(layer_next)

    def _update_gradients(self, x: np.array, reg_lambda: float):
        """Compute and update gradients of each layers.

        Args:
            x: feature matrix
            reg_lambda: regularization factor

        Returns:

        """

        for i, layer in enumerate(self.layers):
            if i == 0:
                a_in = x
            else:
                prev_layer = self.layers[i - 1]
                a_in = prev_layer.A
            layer.update_gradients(a_in, reg_lambda)

    def save(self, path: str):
        """Save model to json."""

        model_dict = self.return_model_dict()
        model_utils.save_json(model_dict, path)

    def load(self, path: str):
        """Load model from json file."""

        model_dict = model_utils.read_json(path)
        self.__init__(model_dict=model_dict)

    def _build_architecture_from_dict(self):
        """Build architecture of MLP from dict Instantiates Dense layers inside of a
        list.
        """

        self.layers = []
        for layer_dict in self._model_dict["layers"]:
            dense = Dense(layer_dict=layer_dict)
            self.layers.append(dense)
        self.n_layers = len(self.layers)

    def return_model_dict(self) -> dict:
        """Returns model information as a json.

        Returns: model info

        """

        model_dict = {"layers": self._get_layers()}

        return model_dict

    def _get_layers(self) -> List[Layer]:
        """Return layer weights and activation type in a list of dicts.

        Returns:
            list of layers

        """

        layers = []
        for layer in self.layers:
            layer_i = layer.to_dict()
            layers.append(layer_i)

        return layers

    def reset_layers(self):
        """Resets layrers of model."""
        for layer in self.layers:
            layer.reset_layer()

    def plot_train(self):
        """Plot results of train operation"""
        if self.train_log is not None:
            train_log = self.train_log.copy()
            train_log["batch"] = train_log.index
            train_log.plot(x="batch", y="loss")
