import logging
import numpy as np
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer


class HiddenLayer(BaseLayer):

    logger = logging.getLogger("HiddenLayer")

    def __init__(self, name, n_in, n_out, activation=None):
        self.logger.debug("Creating hidden layer {0}".format(name))

        self.activation = activation if activation is not None else 'tanh'

        self.logger.debug("Activation {0} - {1} inputs and {2} outputs".format(self.activation, n_in, n_out))

        self.w = create_shared_variable(name + "_w", (n_in, n_out), activation)
        self.b = create_shared_variable(name + "_b", n_out, 0)
        self.params = [self.w,  self.b]

        self.input_shape = n_in
        self.output_shape = n_out

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


