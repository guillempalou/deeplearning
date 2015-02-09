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
    def transform(self, x, mode='train'):
        return "No method defined"

    @abc.abstractmethod
    def definition(self):
        return "No method defined"

    def drop_output(self, x):
        rng = np.random.RandomState()
        srng = T.shared_randomstreams.RandomStreams(rng.randint(1e5))
        mask = srng.binomial(n=1, p=1 - self.dropout, size=x.shape)
        return x * T.cast(mask, theano.config.floatX)

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

