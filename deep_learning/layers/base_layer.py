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
    def definition(self):
        return "No method defined"

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

    def __str__(self):
        return self.definition()

