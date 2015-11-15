from numpy.testing import assert_almost_equal
import numpy as np

from deep_learning.nets.logistic import LogisticNet
import theano
import theano.tensor as T

from deep_learning.nets.mlp import MLP


def test_logistic():
    net = LogisticNet(name="logistic", in_shape=2, out_shape=2)
    x = T.fvector('x')
    f = theano.function([x], net.transform(x))
    net.set_paramaters_values({
        "logistic_output_w": [[1, 1], [1, 1]],
        "logistic_output_b": [1, 1]
    })
    assert_almost_equal([[0.5, 0.5]], f([1, 1]))


def test_mlp():
    net = MLP(name="mlp", in_shape=4, hidden_shape=3, out_shape=2)
    x = T.fvector('x')
    f = theano.function([x], net.transform(x))
    net.set_paramaters_values({x[0]: np.ones_like(x[1]) for x in net.get_parameters_values().items()})
    assert_almost_equal([[0.5, 0.5]], f([1, 1, 1, 1]))
