import networkx as nx

from deep_learning.nets.feed_forward_net import FeedForwardNet


def create_forward_net(name, layers):

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
        

    net = FeedForwardNet(name=name, in_shape=in_shape, out_shape=out_shape)
    net.create_net(g)
    return net

