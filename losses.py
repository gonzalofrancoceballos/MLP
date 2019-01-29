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


class MSE():
    """
    Class that implements Mean Squared Error
    """
    def forward(self, actual, prediction):
        """
        Compute MSE error between targt and prediction
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise MSE 
        """
        return 0.5*((prediction-actual)**2) 
    
    def derivate(self, actual, prediction):
        """
        Compute the derivative of MSE error 
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of MSE 
        """
        return prediction - actual
    
class Logloss():
    """
    Class that implements Logloss Error
    """
    def forward(self, actual, prediction):
        """
        Compute Logloss error between targt and prediction
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise Logloss
        """
        return actual*np.log(prediction) + (1-actual)*np.log(1-prediction)
    
    def derivate(self, actual, prediction):
        """
        Compute the derivative of Logloss error 
        :param actual: target vector (type: np.array)
        :param actual: predictions vector (type: np.array)
        :return: vector containing element-wise derivative of Logloss
        """
        return (actual-prediction)/(prediction*(1-actual))