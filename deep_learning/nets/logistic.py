import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy
from deep_learning.common.functions import l2_sqnorm
from deep_learning.common.functions import l1_norm
from deep_learning.common.tensors import create_theano_tensor
from deep_learning.layers.softmax import SoftMaxLayer
from deep_learning.nets.base_net import BaseNet

theano.config.optimizer = 'None'


class Logistic(BaseNet):
    def __init__(self, name, n_in, n_out, learning_rate=0.001, l1=0, l2=0):
        self.name = name
        self.output_layer = SoftMaxLayer(name + "_output", n_in, n_out)
        self.layers = [self.output_layer]

        # define the learning parameters
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2

        # define the parameters
        self.params = [self.output_layer.w, self.output_layer.b]
        self.train_function = None

        # define inputs and outputs based on the layers
        self.X = create_theano_tensor('logistic_gt', 2, float)
        self.Y = create_theano_tensor('logistic_gt', 1, int)

        self.cost = self.cost_function(self.X, self.Y)

        # generate the prediction function
        self.predict = theano.function(inputs=[self.X],
                                       outputs=self.transform(self.X),
                                       allow_input_downcast=True)

        # call the base class
        super(Logistic, self).__init__(name)

    def transform(self, x):
        return self.output_layer.transform(x)

    def gradient(self, layer=None):
        return T.grad(cost=self.cost, wrt=self.params)

    def update_parameters(self):
        gradient = self.gradient()
        return [[p, p - gradient[i] * self.learning_rate] for i, p in enumerate(self.params)]

    def cost_function(self, x, y):
        # TODO add more costs functions
        nlog = -T.sum(T.log((self.transform(x))[T.arange(y.shape[0]), y]))
        return nlog + \
               self.l1 * l1_norm(self.params) + \
               self.l2 * l2_sqnorm(self.params)


    def train(self, x=None, y=None, index=0):
        self.train_function(x, y)

