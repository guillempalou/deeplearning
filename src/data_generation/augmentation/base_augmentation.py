import abc
import logging
import numpy as np

class BaseAugmentation(object):

    logger = logging.getLogger("BaseAugmentation")

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.rng = np.random.RandomState()
        pass

    def gaussian_noise(self, x, std):
        return x + self.rng.normal(0, std, x.shape)

    def salt_and_pepper_noise(self, x, p):
        noise = self.rng.uniform(0, 1)

        salt = (noise > ((1-p)/2)).astype(np.float32)
        pepper = (noise < p/2).astype(np.float32)

        x[np.nonzero(salt)] = 1
        x[np.nonzero(pepper)] = 0

        return x

    @abc.abstractmethod
    def augment(self, x, y):
        pass
