import logging

import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from deep_learning.layers.base_layer import BaseLayer
from deep_learning.units.activation.base_activation import BaseActivation


class Convolutional2DLayer(BaseLayer):
    logger = logging.getLogger("Convolutional2DLayer")

    def __init__(self, **kwargs):

        super(Convolutional2DLayer, self).__init__(**kwargs)

        self.logger.debug("Creating convolutional layer {0}".format(self.name))

        self.logger.debug("Initializing weights and biases")
        self.w = kwargs["initializer"]["w"].create_shared()
        self.b = kwargs["initializer"]["b"].create_shared()

        self.stride = kwargs.get("stride", (1, 1))
        self.pool = kwargs.get("pool", (1, 1))

        # our activation function is the identity by default
        self.activation = kwargs.get("activation", BaseActivation())

        # setup convolutional parameters
        self.filter = kwargs["filter"]
        self.n_filters = self.out_shape[0]
        self.n_channels = self.in_shape[0]

        self.filter_shape = [self.n_filters, self.n_channels, self.filter[0], self.filter[1]]

        self.logger.debug("Input shape {0} - Activation {1}".format(self.in_shape, self.activation))
        self.logger.debug("Filters {0} - Stride {1}".format(self.filter_shape, self.stride))
        self.logger.debug("Output shape {0}".format(self.out_shape))

    def transform(self, x, mode='train'):
        """
        Convolve the input matrix
        :param x: input
        :param mode:
        :return:
        """
        conv_output = conv.conv2d(input=x,
                                  filters=self.w,
                                  subsample=self.stride,
                                  filter_shape=self.filter_shape,
                                  image_shape=self.in_shape)

        pool = downsample.max_pool_2d(conv_output, ds=self.pool, ignore_border=True)

        return self.activation(pool + self.b.dimshuffle('x', 0, 'x', 'x'))

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

    def get_parameters(self):
        """
        Returns a dictionary of the parameters
        :return:
        """
        self.logger.debug("Getting parameters")
        self.logger.debug("Are {0} and {1}".format(self.w.name, self.b.name))
        return {self.w.name: self.w, self.b.name: self.b}
