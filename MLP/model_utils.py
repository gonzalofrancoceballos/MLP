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