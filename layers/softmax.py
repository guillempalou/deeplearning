import numpy as np
import theano
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer

theano.config.exception_verbosity = 'high'


class SoftMaxLayer(BaseLayer):

    def __init__(self, name, n_in, n_out):
        self.w = create_shared_variable(name + "_w", (n_in, n_out), 'random')
        self.b = create_shared_variable(name + "_b", n_out, 'random')
        self.params = [self.w,  self.b]
        super(SoftMaxLayer, self).__init__(name,
                                           np.asarray([n_in]),
                                           np.asarray([n_out]), truth_type=int)

    def transform(self, x):
        return T.nnet.softmax(T.dot(x, self.w) + self.b)

    def get_weights(self):
        return self.w.get_value()

    def get_bias(self):
        return self.b.get_value()

    def get_parameters(self):
        return [self.w, self.b]