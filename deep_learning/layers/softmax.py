import logging
import numpy as np
import theano
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer

theano.config.exception_verbosity = 'high'


class SoftMaxLayer(BaseLayer):

    logger = logging.getLogger("SoftMaxLayer")

    def __init__(self, name, n_in, n_out):
        self.logger.debug("Creating softmax {0}".format(name))
        self.logger.debug("Layer with {0} inputs and {1} outputs".format(n_in, n_out))

        self.w = create_shared_variable(name + "_w", (n_in, n_out), 'random')
        self.b = create_shared_variable(name + "_b", n_out, 0)
        self.params = [self.w,  self.b]

        self.input_shape = n_in
        self.output_shape = n_out

        super(SoftMaxLayer, self).__init__(name)

    def transform(self, x):
        return T.nnet.softmax(T.dot(x, self.w) + self.b)

    def get_weights(self):
        return self.w.get_value()

    def get_bias(self):
        return self.b.get_value()

    def get_parameters(self):
        return [self.w, self.b]