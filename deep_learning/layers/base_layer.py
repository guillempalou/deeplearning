import abc
import numpy as np
import theano
import theano.tensor as T
from deep_learning.common.tensors import create_theano_tensor


class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def transform(self, x):
        return "No method defined"

    @abc.abstractmethod
    def get_parameters(self):
        return self.params

    @abc.abstractmethod
    def get_weights(self):
        return self.w.get_value()

    @abc.abstractmethod
    def get_bias(self):
        return self.b.get_value()

    def l1_norm(self):
        c = 0
        for p in self.params:
            c += T.sum(abs(p))
        return c

    def l2_sqnorm(self):
        sq = 0
        for p in self.params:
            sq += T.sum(T.square(p))
        return sq
