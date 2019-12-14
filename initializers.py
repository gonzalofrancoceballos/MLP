import numpy as np
from abc import abstractmethod


class Initializer:
    @abstractmethod
    def initialize(self, layer) -> tuple:
        pass


class Rand(Initializer):
    def initialize(self, layer) -> tuple:
        W = np.random.rand(layer.input_dim, layer.output_dim)
        b = np.random.rand(1, layer.output_dim)
        dW = np.zeros([layer.input_dim, layer.output_dim])
        db = np.zeros([1, layer.output_dim])
        return W, b, dW, db


class Glorot(Initializer):
    def initialize(self, layer) -> tuple:
        if layer.activation.type == "relu":
            W = np.random.normal(0, size=[layer.input_dim, layer.output_dim]) * np.sqrt(
                2 / layer.input_dim
            )
        else:
            W = np.random.normal(0, size=[layer.input_dim, layer.output_dim]) * np.sqrt(
                1 / layer.input_dim
            )

        b = np.zeros([1, layer.output_dim])
        dW = np.zeros([layer.input_dim, layer.output_dim])
        db = np.zeros([1, layer.output_dim])

        return W, b, dW, db
