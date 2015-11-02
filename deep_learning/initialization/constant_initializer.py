import logging
import numpy as np
import theano
from deep_learning.initialization.base_initializer import BaseInitializer


class ConstantInitializer(BaseInitializer):
    logger = logging.getLogger(__name__ + "." + "ConstantInitializer")

    def __init__(self):
        super(ConstantInitializer, self).__init__()
        pass

    def __call__(self, **kwargs):
        shape = kwargs["shape"]
        type = np.float32 if "type" not in kwargs else kwargs["type"]
        value = kwargs["value"]
        return theano.shared(np.ones(shape)*value, type=type)