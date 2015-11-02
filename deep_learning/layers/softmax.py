import logging
import theano.tensor as T
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer


class SoftMaxLayer(BaseLayer):
    logger = logging.getLogger("SoftMaxLayer")

    def __init__(self, **kwargs):
        super(SoftMaxLayer, self).__init__(**kwargs)

        self.logger.debug("Creating softmax {0}".format(self.name))
        self.logger.debug("Layer with {0} inputs and {1} outputs".format(self.in_shape, self.out_shape))

        self.w = create_shared_variable(self.name + "_w", (self.in_shape, self.out_shape), self.init_weights)
        self.b = create_shared_variable(self.name + "_b", self.out_shape, self.init_weights)
        self.params = [self.w, self.b]

    def transform(self, x, mode):
        return T.nnet.softmax(T.dot(x, self.w) + self.b)

    def get_weights(self):
        return self.w.get_value()

    def get_bias(self):
        return self.b.get_value()

    def get_variables(self):
        return [self.w, self.b]