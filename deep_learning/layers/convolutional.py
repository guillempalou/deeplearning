import logging
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer


class ConvolutionalLayer(BaseLayer):

    logger = logging.getLogger("ConvolutionalLayer")

    def definition(self):
        return {
            'name': self.name,
            'type': 'convolutional',
            'input_shape': list(self.input_shape),
            'filters': self.filters,
            'output_shape': list(self.output_shape),
            'activation': self.activation,
            'dropout': self.dropout,
            'pool': self.pool,
            'stride': self.stride
        }


    def __init__(self, name, input,
                 filters, stride=(1, 1), pool=(1, 1),
                 activation='relu', dropout=0,
                 init_bias=None,
                 init_weights=None):
        """
            Initializes 2d convolutional layer.
            We input 3D tensors, but convert internally to 4D

        :param dropout:
            :param name: name of the layer
            :param input: 3D tensor (input maps, image rows, image cols)
            :param filters: 3D tensor (output maps, kernel rows, kernel cols)
            :param stride: 2D matrix of filter subsampling
            :param activation: type of activation
            :return:
            """
        self.logger.debug("Creating hidden layer {0}".format(name))

        self.input_shape = np.array(input)
        self.filters = filters
        self.filter_shape = np.array((filters[0], input[0], filters[1], filters[2]))
        self.pool = pool
        self.activation = activation
        self.stride = stride
        self.dropout = dropout
        self.init_weights = self.activation if init_weights is None else init_weights
        self.init_bias = 0 if init_bias is None else init_bias

        out_rows = (self.input_shape[1] - self.filter_shape[2] + 1)
        out_cols = (self.input_shape[2] - self.filter_shape[3] + 1)

        valid_rows = int(np.floor(np.ceil(out_rows * 1.0 / self.stride[0]) / self.pool[0]))
        valid_cols = int(np.floor(np.ceil(out_cols * 1.0 / self.stride[1]) / self.pool[1]))

        self.output_shape = np.array((self.filter_shape[0],
                                      valid_rows,
                                      valid_cols))

        self.w = create_shared_variable(name + "_w", self.filter_shape, self.init_weights)
        self.b = create_shared_variable(name + "_b", self.filter_shape[0], self.init_bias)
        self.params = [self.w, self.b]

        self.logger.debug("Input shape {0} - Activation {1}".format(self.input_shape, self.activation))
        self.logger.debug("Filters {0} - Stride {1}".format(self.filter_shape, self.stride))
        self.logger.debug("Output shape {0}".format(self.output_shape))
        self.logger.debug("Dropout probability {0}".format(self.dropout))

        # call the super constructor to initialize all variables
        super(ConvolutionalLayer, self).__init__(name)


    def transform(self, x, mode='train'):

        conv2d_input = None
        conv_output = conv.conv2d(input=x,
                                  filters=self.w,
                                  subsample=self.stride,
                                  filter_shape=self.filter_shape,
                                  image_shape=conv2d_input)

        pool = downsample.max_pool_2d(conv_output, ds=self.pool, ignore_border=True)

        r = None
        if self.activation == 'tanh':
            r =  T.tanh(pool + self.b.dimshuffle('x', 0, 'x', 'x'))

        if self.activation == 'relu':
            r = pool + self.b.dimshuffle('x', 0, 'x', 'x')
            r = r*(r > 0.)

        if self.activation == 'linear':
            r = pool + self.b.dimshuffle('x', 0, 'x', 'x')

        if mode == 'test':
            return r

        return self.drop_output(r / (1-self.dropout))
