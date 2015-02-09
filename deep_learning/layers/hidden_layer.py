import logging
import numpy as np
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer


class HiddenLayer(BaseLayer):

    logger = logging.getLogger("HiddenLayer")

    def definition(self):
        return {
            'name': self.name,
            'type': 'hidden',
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'activation': self.activation,
            'dropout': self.dropout
        }

    def __init__(self, name, n_in, n_out, activation, init_weights, init_bias, dropout):
        self.logger.debug("Creating hidden layer {0}".format(name))

        self.activation = activation if activation is not None else 'tanh'

        self.init_weights = self.activation if init_weights is None else init_weights
        self.init_bias = 0 if init_bias is None else init_bias

        self.w = create_shared_variable(name + "_w", (n_in, n_out), self.init_weights)
        self.b = create_shared_variable(name + "_b", n_out, self.init_bias)
        self.params = [self.w,  self.b]

        self.input_shape = n_in
        self.output_shape = n_out
        self.dropout = dropout

        self.logger.debug("Activation {0} - {1} inputs and {2} outputs".format(self.activation, n_in, n_out))
        self.logger.debug("Dropout probability {0}".format(self.dropout))

        super(HiddenLayer, self).__init__(name)

    def transform(self, x, mode='train'):
        xw = T.dot(x, self.w) + self.b
        if self.activation == 'tanh':
            xw = T.tanh(xw)

        if self.activation == 'relu':
            xw *= xw > 0.

        if mode == 'test':
            return xw

        return self.drop_output(xw/(1-self.dropout))

    def get_bias(self):
        return self.b.get_value()

    def get_weights(self):
        return self.w.get_value()

    def get_parameters(self):
        return [self.w, self.b]


