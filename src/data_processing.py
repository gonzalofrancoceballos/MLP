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
from typing import List, Union
from .tensor import Tensor


class Batcher:
    """Batcher class. Given a list of np.arrays of same 0-dimension, returns a
    a list of batches for these elements.
    """

    def __init__(
        self,
        data: Union[List[Tensor], Tensor],
        batch_size: int,
        shuffle_on_reset: bool = False,
    ):
        """

        Args:
            data: list containing np.arrays
            batch_size: size of each batch
            shuffle_on_reset: flag to shuffle data
        """

        self.data = data
        self.batch_size = batch_size
        self.shuffle_on_reset = shuffle_on_reset

        if type(data) == list:
            self.data_size = data[0].shape[0]
        else:
            self.data_size = data.shape[0]
        self.n_batches = int(np.ceil(self.data_size / self.batch_size))
        self.idx = np.arange(0, self.data_size, dtype=int)
        if shuffle_on_reset:
            np.random.shuffle(self.idx)
        self.current = 0

    def shuffle(self):
        """Re-shufle the data."""
        np.random.shuffle(self.idx)

    def reset(self):
        """Reset iteration counter."""

        if self.shuffle_on_reset:
            self.shuffle()
        self.current = 0

    def next(self):
        """Get next batch.

        Returns:
            list of np.arrays

        """

        i_select = self.idx[
            (self.current * self.batch_size) : ((self.current + 1) * self.batch_size)
        ]
        batch = []
        for elem in self.data:
            batch.append(elem[i_select])

        if self.current < (self.n_batches - 1):
            self.current = self.current + 1
        else:
            self.reset()

        return batch
