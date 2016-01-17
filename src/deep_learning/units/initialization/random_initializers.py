import logging

import numpy as np
import scipy.stats as scs
import theano

from deep_learning.units.initialization.base_initializer import BaseInitializer


class RandomInitializer(BaseInitializer):
    """
    Returns a shared variable with random coefficients
    according to a certain distribution. Distribution is a
    scipy.stats distribution
    """
    logger = logging.getLogger(__name__ + "." + "RandomInitializer")

    def __init__(self, name, shape, distribution):
        super(RandomInitializer, self).__init__(name, shape)
        self.distribution = distribution
        self.shape = shape

    def create_shared(self, **kwargs):
        return theano.shared(self.distribution.rvs(self.shape), name=self.name, **kwargs)


class FanInOutInitializer(RandomInitializer):
    """
    Initializes a shared variable according to [INSERT REF]
    """
    logger = logging.getLogger("NormalizedRandomInitializer")

    def __init__(self, name, shape):
        self.logger.debug("Initializing {0} with shape {1}".format(name, shape))
        # the shape is a 4D tensor,
        fan_in = shape[0] if shape.ndim == 2 else np.prod(shape[1:])
        fan_out = shape[1] if shape.ndim == 2 else shape[0] * np.prod(shape[2:])
        # TODO check the formula
        bound = np.sqrt(6. / (fan_in + fan_out))
        super(FanInOutInitializer, self).__init__(name, shape,
                                                  scs.uniform(-bound, 2*bound))
