import abc
import logging

import theano
import numpy as np
import theano.tensor as T

from deep_learning.common.functions import l1_norm, l2_sqnorm
from deep_learning.common.tensors import create_theano_tensor


class BaseNet(object):
    logger = logging.getLogger("BaseNet")

    def __init__(self, name):
        self.name = name
        self.cost = None

        # X and Y should be defined in a derived class

        # predefine the test and train functions
        self.index = None
        self.train_function = None
        self.train_function_raw = None

        self.test_function = None
        self.test_function_raw = None

        # theano functions variables
        self.test_cost = None
        self.train_cost = None

        # setup shared variables if needed
        self.train_set_x = None
        self.train_set_y = None

        # define the predict function
        self.predict = theano.function(inputs=[self.X],
                                       outputs=self.transform(self.X),
                                       allow_input_downcast=True)


    @abc.abstractmethod
    def transform(self, x):
        pass

    def gradient(self, cost):
        return T.grad(cost=cost, wrt=self.params)

    def update_parameters(self, cost, parameters):
        gradient = self.gradient(cost)

        l = parameters.learning_rate
        if parameters.momentum > 0:
            updates = []
            m = parameters.momentum
            for i, param in enumerate(self.params):
                init = np.zeros(param.get_value().shape, dtype=np.float32)
                param_update = theano.shared(init, name=param.name + "_update",
                                             broadcastable=param.broadcastable, borrow=True)

                updates.append([param_update, m * param_update + (1. - m) * gradient[i]])
                updates.append([param, param - l * param_update])
            return updates
        else:
            return [[p, p - l * gradient[i]] for i, p in enumerate(self.params)]

    def train_cost_function(self, x, y, parameters):

        # TODO add more costs functions
        loss = None
        if parameters.train_loss == 'crossentropy':
            loss = -T.mean(T.log(self.transform(x)[T.arange(y.shape[0]), y]))

        if parameters.train_loss == 'mse':
            loss = T.mean(T.sum(T.square(self.transform(x) - y)))


        # TODO add regularization depending on the layer
        return loss + \
               parameters.l1 * l1_norm(self.params) + \
               parameters.l2 * l2_sqnorm(self.params)

    def test_cost_function(self, x, y, parameters):

        if parameters.test_loss == 'accuracy':
            z = T.mean(T.neq(T.argmax(self.transform(x), axis=1), y))

        if parameters.test_loss == 'crossentropy':
            z = -T.mean(T.log(self.transform(x)[T.arange(y.shape[0]), y]))

        return z

    def begin_training(self, parameters, x=None, y=None):

        # train with minibatch. we should transfer to the GPU for speed
        data_in_gpu = False
        if x is not None and y is not None and parameters.minibatch > 1:
            data_in_gpu = True

        if data_in_gpu:
            self.transfer_to_gpu(x, y)

        self.setup_test_functions(parameters, data_in_gpu, x=x, y=y)
        self.setup_train_functions(parameters, data_in_gpu, x=x, y=y)


    def transfer_to_gpu(self, x, y):
        self.logger.debug("Setting up shared datasets")

        # create shared variables for x and y
        self.index = create_theano_tensor(self.name + "_index", 0, int)
        self.train_set_x = theano.shared(x, name=self.name + "_set_x",
                                         allow_downcast=True, borrow=True)

        self.train_set_y = theano.shared(y, name=self.name + "_set_y",
                                         allow_downcast=True, borrow=True)


    def setup_train_functions(self, parameters, data_in_gpu=True, x=None, y=None):
        # train with minibatch. we should transfer to the GPU for speed

        self.logger.debug("Compiling train cost function")
        self.train_cost = self.train_cost_function(self.X, self.Y, parameters)
        self.logger.debug("Compiling updates function")
        updates = self.update_parameters(self.train_cost, parameters)

        if data_in_gpu:
            minibatch = parameters.minibatch

            self.logger.debug("Setting up functions for minibatches in GPU")
            self.train_function = theano.function(inputs=[self.index],
                                                  outputs=[self.train_cost, self.test_cost], updates=updates,
                                                  allow_input_downcast=True,
                                                  givens={
                                                      self.X: self.train_set_x[
                                                              self.index * minibatch: (self.index + 1) * minibatch],
                                                      self.Y: self.train_set_y[
                                                              self.index * minibatch: (self.index + 1) * minibatch]
                                                  })

        self.logger.debug("Setting up functions for raw inputs")
        self.train_function_raw = theano.function(inputs=[self.X, self.Y],
                                                  outputs=[self.train_cost, self.test_cost], updates=updates,
                                                  allow_input_downcast=True)


    def setup_test_functions(self, parameters, data_in_gpu=True, x=None, y=None):
        self.logger.debug("Compiling test cost function")
        self.test_cost = self.test_cost_function(self.X, self.Y, parameters)

        if data_in_gpu:
            minibatch = parameters.minibatch
            self.test_function = theano.function(inputs=[self.index],
                                                 outputs=self.test_cost,
                                                 allow_input_downcast=True,
                                                 givens={
                                                     self.X: self.train_set_x[
                                                             self.index * minibatch: (self.index + 1) * minibatch],
                                                     self.Y: self.train_set_y[
                                                             self.index * minibatch: (self.index + 1) * minibatch]
                                                 })

        self.logger.debug("Setting up test function for raw inputs")
        self.test_function_raw = theano.function(inputs=[self.X, self.Y],
                                                 outputs=self.test_cost,
                                                 allow_input_downcast=True)


    def train(self, x=None, y=None, index=0):
        if x is None and y is None:
            return self.train_function(index)
        else:
            return self.train_function_raw(x, y)


    def test(self, x=None, y=None, index=0):
        if x is None and y is None:
            return self.test_function(index)
        else:
            return self.test_function_raw(x, y)


    def get_parameters_values(self):
        return [p.get_value() for p in self.params]


    def set_paramaters_values(self, values):
        for v, p in zip(values, self.params):
            p.set_value(v)


    def definition(self):
        return [layer.definition() for layer in self.layers]


    def __str__(self):
        return self.definition()