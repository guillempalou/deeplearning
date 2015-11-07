import logging

from src.deep_learning import create_theano_tensor
from src.deep_learning import ConvolutionalLayer
from src.deep_learning import BaseNet


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
        self.N = 0
        self.input_shape = None
        self.output_shape = None
        for layer in layers:
            self.layers.append(layer)
            self.params += self.layers[self.N].params
            self.N += 1

        # define inputs and outputs based on the layers
        # TODO modify the input and output dimensions for different types of nets
        self.X = create_theano_tensor(name + "_X", 4, float)
        self.Y = create_theano_tensor(name + "_Y", 1, int)

        self.input_shape = layers[0].input_shape
        self.output_shape = layers[self.N-1].output_shape


        super(DeepNet, self).__init__(name)

    def transform(self, x, mode='train'):
        input = x
        output = None
        for layer in self.layers:
            if type(layer) != ConvolutionalLayer:
                input = input.flatten(2)

            output = layer.transform(input, mode)
            input = output

        return output

