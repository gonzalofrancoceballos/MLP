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

import model_utils
from layers import Dense
from base import Layer, Model
from train import ModelTrain
from optimizers import Adam


"""
TODO:
- Save train log
- Keep trainlog
"""


class BasicMLP(Model):
    """
    Multi-layer perceptron
    """

    def __init__(self, model_dict: dict = None):
        """
        Initialize network

        :param model_dict: python dictionary containing all necessary information to
        instantiate an existing model (type: dict)
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
            repr_str = "SimpleMLP model:\n"
            for layer in self.layers:
                repr_str = repr_str + str(layer)

            return repr_str
        else:
            return "Empty SimpleMLP model"

    def __str__(self):
        if self.layers:
            repr_str = ""
            for layer in self.layers:
                repr_str = repr_str + str(layer)

            return repr_str
        else:
            return "Empty SimpleMLP model"

    def add(self, layer: Layer):
        """
        Adds a layer to the model

        :param layer: layer to be added (type: Layer)
        """

        if len(self.layers) == 0:
            if layer.input_dim is None:
                raise AttributeError("It is necessary to especify input dim for first layer")
        else:
            layer.input_dim = self.layers[-1].output_dim

        self.layers.append(layer)
        self.n_layers += 1

    def compile(self):
        """
        Compiles model before train
        """
        self.reset_layers()

    def predict(self, x: np.array) -> np.array:
        """
        Computes a forward pass and returns prediction

        Note that this operation will not update Z and A of each weight
        :param x: input matrix to the network (type: np.array)

        :return: output of the network (type: np.array)
        """

        pred = self.forward_prop(x, update=False)
        return pred

    def train(self, loss, train_data, optimizer=Adam(), dev_data=None, params=None):
        self._trainer = ModelTrain()
        self._trainer.train(self, loss, train_data, optimizer, dev_data, params)

    def forward_prop(self, x, update=True):
        """
        Computes a forward pass though the architecture of the network

        :param x: input matrix to the network (type: np.array)
        :param update: flag to update latest values through the network (type: bool)

        :return: output of the network (type: np.array)
        """

        A = x
        for layer in self.layers:
            A = layer.forward(A, update=update)

        return A

    def back_prop(self, x, y, loss, reg_lambda=0.01):
        """
        Computes back-propagation pass through the network
        It retrieves output of the final layer, self.layers[-1].A, and back-propagates
        its error through the layers of the network, computing and updating its gradients

        :param x: input matrix to the network (type: np.array)
        :param y: target vector (type: np.array)
        :param loss: loss funtion object (type: Loss)
        :param reg_lambda: regularizatiopn factor for gradients (type: float)
        """

        self._update_deltas(loss, y)
        self._update_gradients(x, reg_lambda)

    def _update_deltas(self, loss, y):
        """
        Starting from last layer, compute and update deltas in reverse order

        :param loss: loss function object (type: Loss)
        :param y: target verctor (type: np.array)
        """

        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                delta = loss.derivate(y, layer.A) * layer.activation.derivate(layer.Z)
                layer.delta = delta
            else:
                layer_next = self.layers[-i]
                layer.update_delta(layer_next)

    def _update_gradients(self, x, reg_lambda):
        """
        Compute and update gradients of each layers

        :param x: feature matrix (type: np.array)
        :param reg_lambda: regularization factor (type: float)
        """

        for i, layer in enumerate(self.layers):
            if i == 0:
                a_in = x
            else:
                prev_layer = self.layers[i - 1]
                a_in = prev_layer.A
            layer.update_gradients(a_in, reg_lambda)

    def save(self, path: str):
        """
        Save model to json

        :param path: path to save model as json
        """

        model_dict = self.return_model_dict()
        model_utils.save_json(model_dict, path)

    def load(self, path: str):
        """
        Load model from json file

        :param path: path to json file containing model info
        """

        model_dict = model_utils.read_json(path)
        self.__init__(model_dict=model_dict)

    def _build_architecture_from_dict(self):
        """
        Build architecture of MLP from dict Instantiates Dense layers inside of a list
        """

        self.layers = []
        for layer_dict in self._model_dict["layers"]:
            dense = Dense(layer_dict=layer_dict)
            self.layers.append(dense)
        self.n_layers = len(self.layers)

    def return_model_dict(self) -> dict:
        """
        Returns model information as a json

        :return: model info (type: dict)
        """

        model_dict = {
            "layers": self._get_layers()
        }

        return model_dict

    def _get_layers(self):
        """
        Return layer weights and activation type in a list of dicts

        :return: list of layers (type: list[dict])
        """

        layers = []
        for layer in self.layers:
            layer_i = layer.to_dict()
            layers.append(layer_i)

        return layers

    def reset_layers(self):
        """
        Resets layrers of model
        """
        for layer in self.layers:
            layer.reset_layer()

