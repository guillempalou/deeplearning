import logging
import numpy as np
import theano
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer

theano.config.exception_verbosity = 'high'


class SoftMaxLayer(BaseLayer):

    logger = logging.getLogger("SoftMaxLayer")

    def definition(self):
        return {
            'name': self.name,
            'type': 'softmax',
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'init_weights': self.init_weights,
            'init_bias': self.init_bias
        }

    def __init__(self, name, n_in, n_out, init_weights, init_bias):
        self.logger.debug("Creating softmax {0}".format(name))
        self.logger.debug("Layer with {0} inputs and {1} outputs".format(n_in, n_out))

        self.init_weights = 'softmax' if init_weights is None else init_weights
        self.init_bias = 0 if init_bias is None else init_bias

        self.w = create_shared_variable(name + "_w", (n_in, n_out), self.init_weights)
        self.b = create_shared_variable(name + "_b", n_out, self.init_weights)
        self.params = [self.w,  self.b]

        self.input_shape = n_in
        self.output_shape = n_out

        super(SoftMaxLayer, self).__init__(name)

    def transform(self, x, mode):
        return T.nnet.softmax(T.dot(x, self.w) + self.b)

    def get_weights(self):
        return self.w.get_value()

    def get_bias(self):
        return self.b.get_value()

    def get_parameters(self):
        return [self.w, self.b]