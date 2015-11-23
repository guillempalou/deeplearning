import logging
import networkx as nx
import theano
from theano.printing import pp
import theano.tensor as T
from deep_learning.nets.base_net import BaseNet


class FeedForwardNet(BaseNet):
    """
    Base Class for feed forward neural nets
    We can specify a graph and the algorithm will infer the net function
    based on the graph topology
    """
    logger = logging.getLogger(__name__ + "." + "FeedForwardNet")

    def __init__(self, **kwargs):
        super(FeedForwardNet, self).__init__(**kwargs)
        self.order = []

    def create_net(self, net_graph):
        """
        Creates the network and computes the order of the layers
        based on the graph
        :param net_graph: NX graph of the topology
        """
        super(FeedForwardNet, self).create_net(net_graph)
        self.order = nx.topological_sort(net_graph)
        self.logger.debug("Order of the layers is: {0}".format(self.order))

    def transform(self, x, **kwargs):
        """
        Gets the transformation function for a feed forward net topology
        :param x:
        :param kwargs:
        """
        # compute the order of layers
        # if order of computation is not set up

        if self.order is None:
            self.logger.error("You need to create the network first")
            return None

        for layer in self.order:
            self.logger.info("Setting up layer transformation: {0}".format(layer))
            if len(self.inputs[layer]) == 0:
                self.logger.debug("Layer {0} is an input layer".format(layer))
                input_tensor = x
            else:
                self.logger.debug("Stacking tensors, multiple inputs for layer {0}".format(layer))
                self.logger.debug("Inputs are: {0}".format([str(x) for x in self.inputs[layer]]))

                # currently only supports stacking vectors
                input_tensor = T.stack(*[self.outputs[x].ravel() for x in self.inputs[layer]])
            self.logger.debug("Input shape tensor {0}".format(input_tensor.shape))

            # TODO a priori check of input shapes
            # check if the input to the next layer is not
            # the same as we are passing
            # if input_tensor.shape != layer.in_shape:
            #     self.logger.warn("Reshaping tensor from to layers {0}".format(layer))
            #     self.logger.warn("Shape now is {0} and should be {1}".format(input_tensor.shape,
            #                                                                  layer.in_shape))
            #     input_tensor = input_tensor.reshape(layer.in_shape)

            self.outputs[layer] = layer.transform(input_tensor, **kwargs)

        self.logger.debug("Transformation setup completed")
        # TODO accept multiple output layers?
        # get the output layer
        return self.outputs[self.order[-1]]
