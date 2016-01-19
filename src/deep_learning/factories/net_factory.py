import logging

import networkx as nx

from deep_learning.factories.layer_factory import create_layer_from_dict
from deep_learning.nets.feed_forward_net import FeedForwardNet

logger = logging.getLogger("NetFactory")


def create_forward_net_from_dict(net_definition):
    """
    Creates a Deep Net using a specifications dictionary
    :param net_definition: dict of layer specs
    :return: net
    """
    name = net_definition["name"]
    layer_graph = nx.DiGraph()

    previous_layer = None
    for layer_definition in net_definition:

        if previous_layer is not None:
            input = previous_layer.out_shape
            layer_definition["input"] = input

        layer = create_layer_from_dict(layer_definition)
        layer_graph.add_edge(previous_layer, layer)
        layer_graph.add_node(layer)
        previous_layer = layer

    return create_forward_net(name, layer_graph)


def create_forward_net(name, layers):
    """
    Creates a deep net with a list or a graphof layers
    :param name: name of the net
    :param layers: list or NX graph
    :return: net
    """
    g = nx.DiGraph()
    in_shape = None
    out_shape = None

    if isinstance(layers, (tuple, list)):
        in_shape = layers[0].in_shape
        out_shape = layers[-1].out_shape
        g.add_nodes_from(layers)
        g.add_edges_from(zip(layers[:-1], layers[1:]))
    elif isinstance(layers, nx.Graph):
        g = layers
        # TODO support more than one input/output
        inputs = filter(lambda node: g.in_degree(node) == 0, g)
        outputs = filter(lambda node: g.out_degree(node) == 0, g)
        in_shape = inputs[0].in_shape
        out_shape = outputs[0].out_shape

    net = FeedForwardNet(name=name, in_shape=in_shape, out_shape=out_shape)
    net.create_net(g)
    return net
