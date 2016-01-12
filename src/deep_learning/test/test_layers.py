import numpy as np
import scipy
import theano
import theano.tensor as T
from numpy.testing import assert_allclose

from deep_learning.layers.convolutional import Convolutional2DLayer
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


def test_convolutional():

    batch = 2

    n_in_channels, n_out_channels = 2, 2
    n_in_rows, n_in_cols = 5, 5

    filter_rows, filter_cols = 3, 3

    n_out_rows = n_in_rows - filter_rows + 1
    n_out_cols = n_in_cols - filter_cols + 1

    parameters = {
        "name": "test_hidden",
        "in_shape": (n_in_channels, n_in_rows, n_in_cols),
        "out_shape": (n_out_channels, n_out_rows, n_out_cols),
        "filter": (filter_rows, filter_rows),
        "initializer": {"w": ConstantInitializer("w", (n_out_channels, n_in_channels, filter_rows, filter_cols), 1),
                        "b": ConstantInitializer("b", n_out_channels, 1)}
    }
    # result is the filter shape * input channels + bias
    val_result = filter_rows * filter_rows * n_in_channels + 1
    filter_output_gt = val_result * np.ones((batch, n_out_channels, n_out_rows, n_out_cols))

    sl = Convolutional2DLayer(**parameters)

    # set up the vectors
    X = T.dtensor4('X')
    f = theano.function([X], sl.transform(X))
    x = np.ones((batch, n_in_channels, n_in_rows, n_in_cols))
    a = f(x)

    # since the coefficients are the same, the filter returns always the same shape
    assert_allclose(a, filter_output_gt)


def test_convolutional_with_pooling():

    batch = 2

    n_in_channels, n_out_channels = 2, 2
    n_in_rows, n_in_cols = 10, 10

    filter_rows, filter_cols = 3, 3
    pool_rows, pool_cols = 2, 2

    n_out_rows = (n_in_rows - filter_rows + 1) // pool_cols
    n_out_cols = (n_in_cols - filter_cols + 1) // pool_rows

    parameters = {
        "name": "test_hidden",
        "in_shape": (n_in_channels, n_in_rows, n_in_cols),
        "out_shape": (n_out_channels, n_out_rows, n_out_rows),
        "filter": (filter_rows, filter_rows),
        "initializer": {"w": ConstantInitializer("w", (n_out_channels, n_in_channels, filter_rows, filter_cols), 1),
                        "b": ConstantInitializer("b", n_out_channels, 1)},
        "pool": (pool_cols, pool_rows)
    }

    # result is the filter shape * input channels + bias
    val_result = filter_rows * filter_rows * n_in_channels + 1
    filter_output_gt = val_result * np.ones((batch, n_out_channels, n_out_rows, n_out_cols))

    sl = Convolutional2DLayer(**parameters)

    # set up the vectors
    X = T.dtensor4('X')
    f = theano.function([X], sl.transform(X))
    x = np.ones((batch, n_in_channels, n_in_rows, n_in_cols))
    a = f(x)

    # since the coefficients are the same, the filter returns always the same shape
    assert_allclose(a, filter_output_gt)