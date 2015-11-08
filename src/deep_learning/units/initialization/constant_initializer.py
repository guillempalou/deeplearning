import logging

import numpy as np
import theano

from deep_learning.units.initialization import BaseInitializer


class ConstantInitializer(BaseInitializer):
    """
    Returns a constant shared variable
    """

    logger = logging.getLogger(__name__ + "." + "ConstantInitializer")

    def __init__(self, name, shape, value):
        super(ConstantInitializer, self).__init__()
        self.name = name
        self.shape = shape
        self.value = value

    def create_shared(self, **kwargs):
        """
        Returns a shared variable with constant values
        :param kwargs: not used
        :return: shared variable
        """
        return theano.shared(np.ones(self.shape) * self.value)
