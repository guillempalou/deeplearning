import networkx as nx

from deep_learning.layers.softmax_layer import SoftMaxLayer
from deep_learning.nets.feed_forward_net import FeedForwardNet
from deep_learning.units.initialization.random_initializers import FanInOutInitializer
import theano
import theano.tensor as T

class LogisticNet(FeedForwardNet):
    """
    Creates a simple net with a simple Softmax Layer
    """

    def __init__(self, name, in_shape, out_shape, **kwargs):
        super(LogisticNet, self).__init__(name=name, in_shape=in_shape, out_shape=out_shape, **kwargs)

        layer_name = name + "_output"

        initializer_w = FanInOutInitializer(layer_name + "_w", shape=(in_shape, out_shape))
        initializer_b = FanInOutInitializer(layer_name + "_b", shape=out_shape)

        output_layer = SoftMaxLayer(name=layer_name,
                                    in_shape=in_shape,
                                    out_shape=out_shape,
                                    initializer={"w": initializer_w, "b": initializer_b})

        # create variables based for input/output
        self.X = T.dmatrix("input")
        self.Y = T.ivector("output")

        # create a graph for the net
        g = nx.DiGraph()
        g.add_node(output_layer)
        self.create_net(g)


