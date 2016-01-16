import numpy as np
import scipy.stats as scs
from numpy.testing import assert_raises, assert_almost_equal

from deep_learning.factories.initializer_factory import create_initializer, ParametersInitializers
from deep_learning.factories.layer_factory import create_softmax_layer, create_hidden_layer
from deep_learning.units.activation.relu_activation import ReLuActivation


def test_constant():
    init = create_initializer("a", 3, initializer=ParametersInitializers.ConstantInitializer, value=1)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), [1, 1, 1])

    init = create_initializer("a", (3, 4), initializer=ParametersInitializers.ConstantInitializer, value=2)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), 2 * np.ones((3, 4)))

    init = create_initializer("a", np.array((2, 3, 2)), initializer=ParametersInitializers.ConstantInitializer, value=3)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), 3 * np.ones((2, 3, 2)))


def test_random():
    np.random.seed(0)
    pdf = scs.uniform(0, 1)
    init = create_initializer("a", 3, initializer=ParametersInitializers.RandomInitializer, distribution=pdf)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), [0.5488135, 0.71518937, 0.60276338])

    init = create_initializer("a", (3, 4), initializer=ParametersInitializers.RandomInitializer, distribution=pdf)
    s = init.create_shared()
    gt = [[0.54488318, 0.4236548, 0.64589411, 0.43758721],
          [0.891773, 0.96366276, 0.38344152, 0.79172504],
          [0.52889492, 0.56804456, 0.92559664, 0.07103606]]
    assert_almost_equal(s.get_value(), gt)

    init = create_initializer("a", np.array((2, 3, 2)), initializer=ParametersInitializers.RandomInitializer,
                              distribution=pdf)
    s = init.create_shared()
    gt = [[[0.0871293, 0.0202184],
           [0.83261985, 0.77815675],
           [0.87001215, 0.97861834]],

          [[0.79915856, 0.46147936],
           [0.78052918, 0.11827443],
           [0.63992102, 0.14335329]]]

    assert_almost_equal(s.get_value(), gt)


def test_faninout():
    np.random.seed(0)
    init = create_initializer("a", 3, initializer=ParametersInitializers.FanInOutInitializer)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), [0.13806544, 0.60864744, 0.29065872])

    init = create_initializer("a", (3, 4), initializer=ParametersInitializers.FanInOutInitializer)
    s = init.create_shared()
    gt = [[0.08310751, -0.14136384, 0.2701434, -0.11556603],
          [0.72542264, 0.85853661, -0.21582437, 0.54016981],
          [0.05350299, 0.12599404, 0.78805184, -0.79428688]]
    assert_almost_equal(s.get_value(), gt)

    init = create_initializer("a", np.array((2, 3, 2)), initializer=ParametersInitializers.FanInOutInitializer)
    s = init.create_shared()
    gt = [[[-0.76448799, -0.8883829],
           [0.61589228, 0.51504622],
           [0.68512937, 0.88622896]],

          [[0.55393402, -0.07132636],
           [0.5194391, -0.70681842],
           [0.25908339, -0.66038139]]]
    assert_almost_equal(s.get_value(), gt)


def test_softmax_factory():
    layer = create_softmax_layer("layer", 3, 2, {"initializer": "constant", "value": 1})
    assert layer.name == "layer"
    assert layer.in_shape == 3
    assert layer.out_shape == 2
    assert_almost_equal(layer.get_weights(), np.ones((3, 2)))
    assert_almost_equal(layer.get_bias(), np.ones(2))

def test_hidden_factory():
    layer = create_hidden_layer("layer", 3, 2, {"initializer": "constant", "value": 1}, activation=ReLuActivation())
    assert layer.name == "layer"
    assert layer.in_shape == 3
    assert layer.out_shape == 2
    assert isinstance(layer.activation, ReLuActivation)
    assert_almost_equal(layer.get_weights(), np.ones((3, 2)))
    assert_almost_equal(layer.get_bias(), np.ones(2))

def test_convolutional_factory():
    pass
