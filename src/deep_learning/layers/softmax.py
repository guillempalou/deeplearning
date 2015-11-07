import logging
import theano.tensor as T

from deep_learning.layers.base_layer import BaseLayer


class SoftMaxLayer(BaseLayer):
    """
    Layer that gets an input N-vector and produces a K-vector that sums to one
    according to the softmax function
    """
    logger = logging.getLogger("SoftMaxLayer")

    def __init__(self, **kwargs):
        super(SoftMaxLayer, self).__init__(**kwargs)

        self.logger.debug("Creating softmax {0}".format(self.name))
        self.logger.debug("Layer with {0} inputs and {1} outputs".format(self.in_shape, self.out_shape))

        self.w = kwargs["initializer"]["w"].create_shared()
        self.b = kwargs["initializer"]["b"].create_shared()
        self.params = [self.w, self.b]

    def transform(self, x, **kwargs):
        """
        Transform the input using softmax
        :param x: input vector
        :param kwargs: additional arguments (ignored)
        :return: vector
        """
        return T.nnet.softmax(T.dot(x, self.w) + self.b)

    def get_weights(self):
        """
        Returns the weights
        :return: vector w
        """
        return self.w.get_value()

    def get_bias(self):
        """
        Returns the layer bias
        :return: vector b
        """
        return self.b.get_value()

    def get_variables(self):
        """
        Returns the shared variables of the layer
        :return: list of shared variables
        """
        return [self.w, self.b]