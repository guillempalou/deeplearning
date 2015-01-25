import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from deep_learning.common.tensors import create_shared_variable
from deep_learning.layers.base_layer import BaseLayer


class ConvolutionalLayer(BaseLayer):

    def __init__(self, name, input, filters,
                 stride = None, pool=None, activation='relu', minibatch=None):
        """
            Initializes 2d convolutional layer.
            We input 3D tensors, but convert internally to 4D

            :param name: name of the layer
            :param input: 3D tensor (input maps, image rows, image cols)
            :param filters: 3D tensor (output maps, input maps, kernel rows, kernel cols)
            :param stride: 2D matrix of filter subsampling
            :param activation: type of activation
            :return:
            """

        self.input_shape = input
        self.filter_shape = np.asarray((filters[0], input[0], filters[1], filters[2]))
        self.pool = pool
        self.activation = activation
        self.minibatch = 1 if minibatch is None else minibatch

        valid_rows = (self.input_shape[1] - self.filter_shape[2] + 1) // self.pool[0]
        valid_cols = (self.input_shape[2] - self.filter_shape[3] + 1) // self.pool[1]

        self.output_shape = np.asarray((self.filter_shape[0],
                                   valid_rows,
                                   valid_cols))

        self.w = create_shared_variable(name + "_w", self.filter_shape, self.activation)
        self.b = create_shared_variable(name + "_b", self.filter_shape[0], 0)
        self.params = [self.w, self.b]

        # optimization for filtering
        self.conv2d_input = np.array([self.minibatch, input[0], input[1], input[2]])

        # call the super constructor to initialize all variables
        super(ConvolutionalLayer, self).__init__(name)


    def transform(self, x):
        conv_output = conv.conv2d(input=x,
                                  filters=self.w,
                                  filter_shape=self.filter_shape,
                                  image_shape=self.conv2d_input)

        pool = downsample.max_pool_2d(conv_output, ds=self.pool)

        if self.activation == 'tanh':
            return T.tanh(pool + self.b.dimshuffle('x', 0, 'x', 'x'))

        if self.activation == 'relu':
            return T.maximum(pool + self.b.dimshuffle('x', 0, 'x', 'x'), 0)

        return conv_output + self.b.dimshuffle('x', 0, 'x', 'x')

    def get_weights(self):
        return self.w

    def get_parameters(self):
        return self.params

    def get_bias(self):
        return self.b
