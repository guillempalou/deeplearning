import logging
import os

import numpy as np
import networkx as nx
import scipy.stats as scs
import yaml
from numpy.testing import assert_raises, assert_almost_equal

from deep_learning.factories.initializer_factory import create_initializer, InitializerType
from deep_learning.factories.layer_factory import create_softmax_layer, create_hidden_layer, \
    create_convolutional_2d_layer
from deep_learning.factories.net_factory import create_forward_net, create_forward_net_from_dict
from deep_learning.layers.convolutional import Convolutional2DLayer
from deep_learning.layers.hidden_layer import HiddenLayer
from deep_learning.layers.softmax_layer import SoftMaxLayer
from deep_learning.test.test_data_utils import test_data_dir
from deep_learning.units.activation.relu_activation import ReLuActivation


def test_constant():
    init = create_initializer("a", 3, initializer=InitializerType.Constant, value=1)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), [1, 1, 1])

    init = create_initializer("a", (3, 4), initializer=InitializerType.Constant, value=2)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), 2 * np.ones((3, 4)))

    init = create_initializer("a", np.array((2, 3, 2)), initializer=InitializerType.Constant, value=3)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), 3 * np.ones((2, 3, 2)))


def test_random():
    np.random.seed(0)
    pdf = scs.uniform(0, 1)
    init = create_initializer("a", 3, initializer=InitializerType.Random, distribution=pdf)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), [0.5488135, 0.71518937, 0.60276338])

    init = create_initializer("a", (3, 4), initializer=InitializerType.Random, distribution=pdf)
    s = init.create_shared()
    gt = [[0.54488318, 0.4236548, 0.64589411, 0.43758721],
          [0.891773, 0.96366276, 0.38344152, 0.79172504],
          [0.52889492, 0.56804456, 0.92559664, 0.07103606]]
    assert_almost_equal(s.get_value(), gt)

    init = create_initializer("a", np.array((2, 3, 2)), initializer=InitializerType.Random,
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
    init = create_initializer("a", 3, initializer=InitializerType.FanInOut)
    s = init.create_shared()
    assert_almost_equal(s.get_value(), [0.1195682, 0.5271041, 0.2517178])

    init = create_initializer("a", (3, 4), initializer=InitializerType.FanInOut)
    s = init.create_shared()
    gt = [[0.08310751, -0.14136384, 0.2701434, -0.11556603],
          [0.72542264, 0.85853661, -0.21582437, 0.54016981],
          [0.05350299, 0.12599404, 0.78805184, -0.79428688]]
    assert_almost_equal(s.get_value(), gt)

    init = create_initializer("a", np.array((2, 3, 2)), initializer=InitializerType.FanInOut)
    s = init.create_shared()
    gt = [[[-0.63961654, -0.74327446],
           [0.51529245, 0.43091859],
           [0.57322036, 0.74147235]],

          [[0.46345445, -0.05967592],
           [0.43459393, -0.59136672],
           [0.21676471, -0.55251471]]]
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
    np.random.seed(0)
    layer = create_convolutional_2d_layer("layer", (2, 5, 5), 3, (3, 3),
                                          {"initializer": "faninout"},
                                          activation=ReLuActivation())
    assert layer.in_shape == (2, 5, 5)
    assert layer.out_shape == (3, 3, 3)
    assert isinstance(layer.activation, ReLuActivation)

    b_gt = [-0.7131034, -0.82961886, 0.37503727]

    w_gt = [[[[0.03564834, 0.15715209, 0.07504776],
              [0.03277804, -0.05575465, 0.106546],
              [-0.04557986, 0.28611055, 0.3386114]],
             [[-0.08512228, 0.21304585, 0.02110187],
              [0.04969272, 0.31081184, -0.31327097],
              [-0.30151813, -0.35038294, 0.24291119]]],
            [[[0.20313697, 0.27021867, 0.34953342],
              [0.21847453, -0.0281315, 0.20486954],
              [-0.27877294, 0.10218387, -0.26045793]],
             [[0.32474026, 0.01595576, -0.06232211],
              [-0.17194427, 0.20027197, -0.03202327],
              [0.04997709, -0.35142624, 0.08590882]]],
            [[[0.08186314, 0.08539652, 0.32406778],
              [0.13278277, -0.10260092, -0.04598536],
              [0.14432942, -0.32116591, 0.12178919]],
             [[0.12461628, -0.21150667, -0.27099392],
              [-0.13479207, -0.09953158, 0.05126447],
              [-0.04483911, 0.35665782, -0.29062538]]]]

    assert_almost_equal(layer.get_weights(), w_gt)
    assert_almost_equal(layer.get_bias(), b_gt)


def test_net_factory_list():
    hidden = create_hidden_layer("hidden", 10, 5,
                                 {"initializer": "constant", "value": 1},
                                 activation=ReLuActivation())

    output = create_softmax_layer("output", 5, 2, {"initializer": "constant", "value": 1})
    net = create_forward_net("net", [hidden, output])
    assert net.in_shape == 10
    assert net.out_shape == 2


def test_net_factory_graph():
    hidden = create_hidden_layer("hidden", 10, 5,
                                 {"initializer": "constant", "value": 1},
                                 activation=ReLuActivation())

    output = create_softmax_layer("output", 5, 2, {"initializer": "constant", "value": 1})

    g = nx.DiGraph()
    g.add_nodes_from([hidden, output])
    g.add_edge(hidden, output)

    net = create_forward_net("net", g)
    assert net.in_shape == 10
    assert net.out_shape == 2


def test_net_from_yaml():
    net_definition = yaml.load(open(os.path.join(test_data_dir, "test_network.yaml")))
    name = "TestNet"
    net = create_forward_net_from_dict(name, net_definition[name])

    assert isinstance(net.order[0], Convolutional2DLayer)
    assert net.order[0].in_shape == (3, 10, 10)
    assert net.order[0].out_shape == (3, 8, 8)

    assert isinstance(net.order[1], Convolutional2DLayer)
    assert net.order[1].in_shape == net.order[0].out_shape
    assert net.order[1].out_shape == (5, 6, 6)

    assert isinstance(net.order[2], HiddenLayer)
    assert net.order[2].in_shape == np.prod(net.order[1].out_shape)
    assert net.order[2].out_shape == 5

    assert isinstance(net.order[3], SoftMaxLayer)
    assert net.order[3].in_shape == net.order[2].out_shape
    assert net.order[3].out_shape == 2

