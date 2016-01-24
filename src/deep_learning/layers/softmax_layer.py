import logging

from theano import tensor as T
from deep_learning.layers.linear_unit_layer import LinearUnitLayer


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

    def transform(self, x, **kwargs):
        # call linear layer ignoring kwargs
        """
        Transform input variable
        :param x: input matrix
        :param kwargs: ignored
        :return: transformed input
        """
        return T.nnet.softmax(super(SoftMaxLayer, self).transform(x))

