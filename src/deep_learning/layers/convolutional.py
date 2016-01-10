import logging

import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from deep_learning.layers.base_layer import BaseLayer
from deep_learning.units.activation.base_activation import BaseActivation


class ConvolutionalLayer(BaseLayer):
    logger = logging.getLogger("ConvolutionalLayer")


    def __init__(self, **kwargs):
        super(ConvolutionalLayer, self).__init__(**kwargs)

        self.logger.debug("Creating convolutional layer {0}".format(self.name))

        self.logger.debug("Initializing weights and biases")
        self.w = kwargs["initializer"]["w"].create_shared()
        self.b = kwargs["initializer"]["b"].create_shared()

        # our activation function is the identity by default
        self.activation = BaseActivation()

        # setup convolutional parameters
        self.filter = kwargs["filter"]
        self.stride = kwargs.get("stride", (1, 1))
        self.stride = kwargs.get("pool", (1, 1))

        self.logger.debug("Input shape {0} - Activation {1}".format(self.in_shape, self.activation))
        self.logger.debug("Filters {0} - Stride {1}".format(self.filter_shape, self.stride))
        self.logger.debug("Output shape {0}".format(self.output_shape))

        out_rows = (self.in_shape[1] - self.filter_shape[2] + 1)
        out_cols = (self.in_shape[2] - self.filter_shape[3] + 1)

        valid_rows = int(np.floor(np.ceil(out_rows * 1.0 / self.stride[0]) / self.pool[0]))
        valid_cols = int(np.floor(np.ceil(out_cols * 1.0 / self.stride[1]) / self.pool[1]))

        self.output_shape = np.array((self.filter_shape[0],
                                      valid_rows,
                                      valid_cols))


    def transform(self, x, mode='train'):

        conv2d_input = None
        conv_output = conv.conv2d(input=x,
                                  filters=self.w,
                                  subsample=self.stride,
                                  filter_shape=self.filter_shape,
                                  image_shape=conv2d_input)

        pool = downsample.max_pool_2d(conv_output, ds=self.pool, ignore_border=True)

        return self.activation(pool + self.b.dimshuffle('x', 0, 'x', 'x'))
