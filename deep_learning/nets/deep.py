import logging
from deep_learning.common.tensors import create_theano_tensor
from deep_learning.layers.convolutional import ConvolutionalLayer
from deep_learning.nets.base_net import BaseNet

class DeepNet(BaseNet):

    logger = logging.getLogger("DeepNet")

    def __init__(self, name, layers):
        """
            :param definition: list of dictionaries definining layers types and dimensions
            :param learn_parameters: dictionary of addic with learning parameters
            :return: nothing
            """
        self.logger.debug("Creating nnet {0} with {1} layers".format(name, len(layers)))
        self.layers = []
        self.params = []
        self.N = len(layers)
        for layer in layers:
            self.params += layer.params
            self.layers.append(layer)

        # define inputs and outputs based on the layers
        # TODO modify the input and output dimensions for different types of nets
        self.X = create_theano_tensor(name + "_X", 4, float)
        self.Y = create_theano_tensor(name + "_Y", 1, int)

        super(DeepNet, self).__init__(name)

    def transform(self, x):
        input = x
        output = None
        for layer in self.layers:
            if type(layer) != ConvolutionalLayer:
                input = input.flatten(2)

            output = layer.transform(input)
            input = output

        return output

