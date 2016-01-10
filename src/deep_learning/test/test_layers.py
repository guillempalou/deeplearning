import numpy as np
import theano
import theano.tensor as T
from numpy.testing import assert_allclose

from deep_learning.layers.hidden_layer import HiddenLayer
from deep_learning.layers.softmax_layer import SoftMaxLayer
from deep_learning.units.initialization.constant_initializer import ConstantInitializer


def test_base():
    pass


def test_softmax():

    parameters = {
        "name": "test_softmax",
        "in_shape": 3,
        "out_shape": 2,
        "initializer": {"w": ConstantInitializer("w", (3, 2), 1),
                        "b": ConstantInitializer("b", 2, 1)}
    }

    sl = SoftMaxLayer(**parameters)

    # set up the vectors
    X = T.dvector('X')
    f = theano.function([X], sl.transform(X))
    a = f([1, 2, 3])

    # since the coefficients are the same, the softmax retuns 0.5,0.5
    assert_allclose(a, np.array([[0.5, 0.5]]))

def test_hidden():

    parameters = {
        "name": "test_hidden",
        "in_shape": 3,
        "out_shape": 2,
        "initializer": {"w": ConstantInitializer("w", (3, 2), 1),
                        "b": ConstantInitializer("b", 2, 1)}
    }

    sl = HiddenLayer(**parameters)

    # set up the vectors
    X = T.dvector('X')
    f = theano.function([X], sl.transform(X))
    a = f([1, 2, 3])

    # since the coefficients are the same, the softmax retuns 0.5,0.5
    assert_allclose(a, np.array([7, 7]))