import logging

from theano import tensor as T
from deep_learning.layers.linear_unit import LinearUnitLayer


class SoftMaxLayer(LinearUnitLayer):
    """
    Layer that gets an input N-vector and produces a K-vector that sums to one
    according to the softmax function
    """
    logger = logging.getLogger("SoftMaxLayer")

    def __init__(self, **kwargs):
        super(SoftMaxLayer, self).__init__(**kwargs)
        self.logger.debug("Creating softmax {0}".format(self.name))
        self.logger.debug("Layer with {0} inputs and {1} outputs".format(self.in_shape, self.out_shape))

        # set the activation function for this layer as the softmax
        self.activation = T.nnet.softmax
