from abc import abstractmethod

import numpy as np


class Activation:
    """
    Base class for activation function of a layer
    """
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def derivate(self, x):
        pass


class Layer:
    """
    Base class for layer
    """
    @abstractmethod
    def reset_layer(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def _from_dict(self, layer_dict: dict):
        pass

    @abstractmethod
    def update_delta(self, next_layer):
        pass

    @abstractmethod
    def update_gradients(self, a_in: np.array, reg_lambda: float):
        pass


class Loss:
    """
    Base class for loss function for prediction error
    """
    @abstractmethod
    def forward(self, actual: np.array, prediction: np.array) -> np.array:
        pass

    @abstractmethod
    def derivate(self, actual: np.array, prediction: np.array) -> np.array:
        pass


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


class Optimizer:
    """
    Base class for optimizer
    """
    @abstractmethod
    def update_weights(self, layers):
        pass

    @abstractmethod
    def initialize_parameters(self, layers):
        pass