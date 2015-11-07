import numpy as np
import theano
import theano.tensor as T

from deep_learning.initialization.constant_initializer import ConstantInitializer
from deep_learning.layers.softmax import SoftMaxLayer
from numpy.testing import assert_allclose


def test_base():
    pass


def test_softmax():
    parameters = {
        "name": "test_softmax",
        "in_shape": 3,
        "out_shape": 2,
        #TODO change the constructor parameters of initializers to make it easy
        "initializer": {"w": ConstantInitializer(.1), "b": ConstantInitializer("b", 2., 1.)}
    }

    sl = SoftMaxLayer(**parameters)

    # set up the vectors
    X = T.dvector('X')
    f = theano.function([X], sl.transform(X))
    a = f([1, 2, 3])

    # since the coefficients are the same, the softmax retuns 0.5,0.5
    assert_allclose(a, np.array([[0.5, 0.5]]))
