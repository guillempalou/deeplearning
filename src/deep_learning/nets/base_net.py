import logging
import networkx as nx
import theano


class BaseNet(object):
    """
    Base class for a neural network. It defines the basic
    """
    logger = logging.getLogger("BaseNet")

    def __init__(self, **kwargs):
        # set some basic variables that all inputs need to have
        self.name = kwargs["name"]
        self.in_shape = kwargs["in_shape"]
        self.out_shape = kwargs["out_shape"]

        # dictionary (string -> theano variable)
        # containing all the net variables
        self.params = {}

        # dictionary (string -> layer) containing each layer
        self.layers = {}

        # dictionary (layer -> values) with each layer ouput
        self.outputs = {}

        # dictionary (layer -> layers) with each layer input
        # useful to compute dependencies
        self.inputs = {}

        # input and output theano tensor types
        # subclasses should define them
        self.X = None
        self.Y = None

        # train and test theano functions
        self.train = None
        self.test = None

    def fit(self, x, y):
        """
        Does a single step on the training. Training function needs to be set up beforehand
        :param x: input vector
        :param y: output target vector
        :return: loss value
        """
        return self.train(x, y)

    def predict(self, x):
        """
        Functions that predicts the output for a given input
        :param x: input vector
        :return:
        """
        if self.test is None:
            self.setup_test_function()
        return self.test(x)

    def setup_train_function(self, loss, updates):
        # TODO instantiate updates here?
        """
        Sets up the theano function in charge of the training
        :param loss: loss function for the target
        :param updates: updates equations for this net's parameters
        :return:
        """
        self.train = theano.function(name="train_{0}".format(self.name),
                                     inputs=[self.X, self.Y],
                                     outputs=loss,
                                     updates=updates)

    def setup_test_function(self):
        """
        Sets up the theano function in charge of the testing
        """
        self.test = theano.function(name="predict_{0}".format(self.name),
                                    inputs=[self.X],
                                    outputs=self.transform(self.X, mode="test"))

    def transform(self, x, **kwargs):
        """
        Gets an input tensor and produces an output. You should subclass
        to the type of net
        :param x:
        :param kwargs:
        """
        raise NotImplementedError("Net transform method not defined")

    def get_parameters_values(self):
        """
        Gets the values of all parameters for the network as a dict
        :return: dict
        """
        return {param[0]: param[1].get_value() for param in self.params.items()}

    def set_paramaters_values(self, values):
        """
        Sets the values for some (or all) parameters from a dict
        :param values: dict containing parameter values
        """
        for (param, value) in values.items():
            self.params[param].set_value(value)

    def get_parameters(self):
        """
        Returns a list with the parameters, without getting the values
        :retrun: list
        """
        return self.params.values()

    def add_layer(self, layer):
        """
        adds a layer to the network
        :param layer:
        """
        self.layers[layer.name] = layer

        # add parameters to the dictionary
        self.params.update(layer.get_parameters())

    def create_net(self, net_graph):
        """
        Assigns the layer graph to this net
        :param net_graph:
        """
        self.topology = net_graph
        self.logger.info("Computing order of the layers")

        for layer in net_graph.nodes_iter():
            self.logger.debug("Adding layer {0}".format(layer))
            self.add_layer(layer)

        self.inputs = {layer: self.topology.predecessors(layer) for layer in self.layers.values()}

    def definition(self):
        """
        Return the definition of the network
        :return: list of definitions
        """
        s = "Network: {0}\n".format(self.name)

        return s + "\n\t".join([str(layer) for layer in self.layers.values()])

    def __str__(self):
        return self.definition()
