import logging

import networkx as nx
import theano
import theano.tensor as T

from deep_learning.layers.hidden_layer import HiddenLayer
from deep_learning.layers.softmax_layer import SoftMaxLayer
from deep_learning.nets.feed_forward_net import FeedForwardNet
from deep_learning.units.initialization.random_initializers import FanInOutInitializer


class MLP(FeedForwardNet):

    """
    Implementation of the multilayer perceptron. Consists of a hidden layer
    and a softmax output layer. It is possible to choose the activation function
    for the hidder layer
    """

    def __init__(self, name, in_shape, hidden_shape, out_shape, **kwargs):

        super(MLP, self).__init__(name=name, in_shape=in_shape, out_shape=out_shape, **kwargs)

        hidden_name = name + "_hidden"
        output_name = name + "_output"

        initializer_h_w = FanInOutInitializer(hidden_name + "_w", shape=(in_shape, hidden_shape))
        initializer_h_b = FanInOutInitializer(hidden_name + "_b", shape=hidden_shape)
        initializer_w = FanInOutInitializer(output_name + "_w", shape=(hidden_shape, out_shape))
        initializer_b = FanInOutInitializer(output_name + "_b", shape=out_shape)

        hidden_layer = HiddenLayer(name=hidden_name,
                                   in_shape=in_shape,
                                   out_shape=hidden_shape,
                                   initializer={"w": initializer_h_w, "b": initializer_h_b},
                                   **kwargs)

        output_layer = SoftMaxLayer(name=output_name,
                                    in_shape=hidden_shape,
                                    out_shape=out_shape,
                                    initializer={"w": initializer_w, "b": initializer_b},
                                    **kwargs)

        # create a graph for the net
        g = nx.DiGraph()
        g.add_node(hidden_layer)
        g.add_node(output_layer)
        g.add_edge(hidden_layer, output_layer)
        self.create_net(g)

