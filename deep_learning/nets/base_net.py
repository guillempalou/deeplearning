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
        self.train_function = None
        self.train_function_raw = None

        self.test_function = None
        self.test_function_raw = None

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

                updates.append([param_update, m*param_update + (1.-m)*gradient[i]])
                updates.append([param, param - l*param_update])
            return updates
        else:
            return [[p, p - l*gradient[i]] for i, p in enumerate(self.params)]


    def train_cost_function(self, x, y, parameters):

        # TODO add more costs functions
        nlog = -T.mean(T.log(self.transform(x)[T.arange(y.shape[0]), y]))

        # TODO add regularization depending on the layer
        return nlog + \
               parameters.l1 * l1_norm(self.params) + \
               parameters.l2 * l2_sqnorm(self.params)

    def test_cost_function(self, x, y, parameters):
        z = T.mean(T.eq(T.argmax(self.transform(x)),y))
        # nlog = -T.mean(T.log(self.transform(x)[T.arange(y.shape[0]), y]))
        return z

    def begin_training(self, parameters, x=None, y=None):

        minibatch = parameters.minibatch

        self.logger.debug("Compiling train cost function")
        train_cost = self.train_cost_function(self.X, self.Y, parameters)
        self.logger.debug("Compiling test cost function")
        test_cost = self.test_cost_function(self.X, self.Y, parameters)
        self.logger.debug("Compiling updates function")
        updates = self.update_parameters(train_cost, parameters)


        # train with minibatch. we should transfer to the GPU for speed
        if x is not None and y is not None and minibatch > 1:
            # create shared variables for x and y
            index = create_theano_tensor(self.name + "_index", 0, int)

            self.logger.debug("Setting up shared datasets")
            train_set_x = theano.shared(x, name=self.name + "_set_x",
                                        allow_downcast=True, borrow=True)

            train_set_y = theano.shared(y, name=self.name + "_set_y",
                                        allow_downcast=True, borrow=True)

            self.logger.debug("Setting up functions for minibatches")
            self.train_function = theano.function(inputs=[index],
                                                  outputs=train_cost, updates=updates,
                                                  allow_input_downcast=True,
                                                  givens={
                                                      self.X: train_set_x[index * minibatch: (index + 1) * minibatch],
                                                      self.Y: train_set_y[index * minibatch: (index + 1) * minibatch]
                                                  })

            self.test_function = theano.function(inputs=[index],
                                                 outputs=test_cost,
                                                 allow_input_downcast=True,
                                                 givens={
                                                     self.X: train_set_x[index * minibatch: (index + 1) * minibatch],
                                                     self.Y: train_set_y[index * minibatch: (index + 1) * minibatch]
                                                 })


        # train without minibatch
        # define functions just in case
        self.logger.debug("Setting up functions for raw inputs")
        self.train_function_raw = theano.function(inputs=[self.X, self.Y],
                                                  outputs=train_cost, updates=updates,
                                                  allow_input_downcast=True)

        self.test_function_raw = theano.function(inputs=[self.X, self.Y],
                                                 outputs=test_cost,
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