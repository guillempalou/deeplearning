import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import theano
import theano.tensor as T
from deep_learning.units.activation.base_activation import BaseActivation
from deep_learning.units.activation.dropout_activation import DropOutActivation
from deep_learning.units.activation.relu_activation import ReLuActivation
from deep_learning.units.activation.tanh_activation import TanhActivation


def test_base_activation():
    act = BaseActivation()
    x = np.array([-1, 0, 1])
    assert_almost_equal(x, act(x))


def test_dropout_activation():
    act = DropOutActivation(dropout=1)
    x = np.array([-1., 0., 1.], dtype=np.float32)
    X = T.vector("X")
    f = theano.function([X], act(X, mode="test"))
    assert_equal(x, f(x))


def test_relu_activation():
    act = ReLuActivation()
    x = np.array([-1, 0, 1])
    assert_equal([0, 0, 1], act(x, mode="test"))


def test_tanh_activation():
    act = TanhActivation()
    x = np.array([-1, 0, 1])
    X = T.dvector()
    f = theano.function([X], act.__call__(X, mode="test"))
    assert_equal(np.tanh([-1, 0, 1]), f(x))
