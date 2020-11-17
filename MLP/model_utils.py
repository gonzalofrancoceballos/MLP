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
import json


def sigmoid(x):
    """
    Sigmoid function
    :param x: input matrix (type:np.array)
    :output: result of applying sigmoid functin element-wise
    """
    return 1.0 / (1.0 + np.exp(-x))


def read_json(file):
    """
    Wrapper function to read json file into a python dictionary
    :param file: path to json file
    :return: content of json file (type: dict)
    """
    with open(file) as f:
        data = json.load(f)
    return data


def save_json(dict_object, file):
    """
    Wraper function to save a dictionary as a json file
    :param dict_object: python dictionary to save
    :param file: path to json file
    """

    with open(file, "w") as fp:
        json.dump(dict_object, fp)


class DummyLogger:
    """
    Dummy print function wrapper that is called like a logger
    """

    def __init__(self):
        """
        Initialize object
        """
        self.info = print
        self.warn = print
        self.warning = print
        self.debug = print
        self.error = print
