from deep_learning.common.functions import l1_norm, l2_sqnorm
from deep_learning.common.tensors import create_theano_tensor
from deep_learning.nets.base_net import BaseNet
import theano
import theano.tensor as T
from deep_learning.nets.net_factory import create_layer


class ConvolutionalNet(BaseNet):

    def __init__(self, definition, learn_parameters, name):
        """
            :param definition: list of dictionaries definining layers types and dimensions
            :param learn_parameters: dictionary of addic with learning parameters
            :return: nothing
            """
        self.layers = []
        self.params = []
        self.N = len(definition)
        for layer_def in definition:
            layer = create_layer(layer_def)
            self.params += layer.params
            self.layers.append(layer)

        # define inputs and outputs based on the layers
        self.X = create_theano_tensor(name + "_X", 4, float)
        self.Y = create_theano_tensor(name + + "_Y", 2, int)
        self.cost = self.cost_function(self.X, self.Y)

        # get all the learning parameters
        self.learning_rate = learn_parameters.learning_rate
        self.l1 = learn_parameters.l1
        self.l2 = learn_parameters.l1

        super(ConvolutionalNet, self).__init__(name)

    def transform(self, x):
        input = x
        output = None
        for layer in self.layers:
            output = layer.transform(input)
            input = output
        return output

    def gradient(self, layer=None):
        return T.grad(cost=self.cost, wrt=self.params)

    def update_parameters(self):
        gradient = self.gradient()
        return [[p, p - gradient[i] * self.learning_rate] for i, p in enumerate(self.params)]

    def cost_function(self, x, y):
        # TODO add more costs functions
        nlog = -T.mean(T.log(self.transform(x)[:, y]))
        return nlog + \
               self.l1 * l1_norm(self.params) + \
               self.l2 * l2_sqnorm(self.params)


    def train(self, x=None, y=None, index=0):
        self.train_function(x, y)