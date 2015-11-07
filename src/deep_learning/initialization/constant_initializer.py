import logging
import numpy as np
import theano

from deep_learning.initialization.base_initializer import BaseInitializer


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
        return theano.shared(np.ones(self.shape) * self.value)
