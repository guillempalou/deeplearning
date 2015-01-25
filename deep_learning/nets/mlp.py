from math import sqrt

import theano
import theano.tensor as T
import numpy as np
from deep_learning.common.functions import l1_norm, l2_sqnorm
from deep_learning.common.tensors import create_theano_tensor
from deep_learning.layers.softmax import SoftMaxLayer
from deep_learning.nets.base_net import BaseNet

from deep_learning.nets.logistic import Logistic
from deep_learning.layers.hidden_layer import HiddenLayer


class MLP(BaseNet):

    def __init__(self, name, n_in, n_hidden, n_out, learning_rate=0.01, l1=0, l2=0):

        self.name = name

        self.hidden_layer = HiddenLayer(name + "_hidden", n_in, n_hidden, activation='tanh')
        self.output_layer = SoftMaxLayer(name + "_output", n_hidden, n_out)

        self.layers = [self.hidden_layer, self.output_layer]

        # TODO remove training parameters from the model
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate

        self.X = create_theano_tensor(name + "_X", 2, float)
        self.Y = create_theano_tensor(name + "_Y", 1, int)

        self.params = self.hidden_layer.params + self.output_layer.params
        self.predict = theano.function(inputs=[self.X],
                                       outputs=self.transform(self.X),
                                       allow_input_downcast=True)

        self.train_function = None
        self.cost = self.cost_function(self.X, self.Y)

        super(MLP, self).__init__(name)


    def transform(self, x):
        output_hidden = self.hidden_layer.transform(x)
        output = self.output_layer.transform(output_hidden)
        return output

    def cost_function(self, x, y):
        nlog = -T.mean(T.log(self.transform(x)[T.arange(y.shape[0]), y]))
        return nlog + \
               self.l1 * l1_norm(self.params) + \
               self.l2 * l2_sqnorm(self.params)

    def gradient(self, layer=None):
        return T.grad(cost=self.cost, wrt=self.params)

    def update_parameters(self):
        gradient = self.gradient()
        return [[p, p - gradient[i] * self.learning_rate] for i, p in enumerate(self.params)]


    def train(self, x=None, y=None, index=0):
        if x is None and y is None:
            self.train_function(index)
        else:
            self.train_function(x, y)

