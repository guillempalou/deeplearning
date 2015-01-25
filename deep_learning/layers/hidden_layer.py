import numpy as np
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer


class HiddenLayer(BaseLayer):

    def __init__(self, name, n_in, n_out, activation=None):
        self.w = create_shared_variable(name + "_w", (n_in, n_out), 'tanh')
        self.b = create_shared_variable(name + "_b", n_out, 0)
        self.params = [self.w,  self.b]
        self.activation = activation
        super(HiddenLayer, self).__init__(name)

    def transform(self, x):
        xw = T.dot(x, self.w) + self.b
        if self.activation == 'tanh':
            return T.tanh(xw)

        if self.activation == 'relu':
            return T.maximum(xw, 0)

        return xw

    def get_bias(self):
        return self.b.get_value()

    def get_weights(self):
        return self.w.get_value()

    def get_parameters(self):
        return [self.w, self.b]


