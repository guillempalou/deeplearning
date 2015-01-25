import abc
import logging
import theano
import numpy as np
import theano.tensor as T
from deep_learning.common.tensors import create_theano_tensor

class BaseNet(object):
    logger = logging.getLogger("BaseNet")

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def cost_function(self, x, y):
        pass

    @abc.abstractmethod
    def gradient(self, layer=None):
        pass

    @abc.abstractmethod
    def update_parameters(self):
        pass

    @abc.abstractmethod
    def train(self, x=None, y=None, index=0):
        pass

    @abc.abstractmethod
    def transform(self, x):
        pass

    # TODO define abstract property for prediction
    # TODO define abstract property for layers

    def get_parameters(self):
        return self.params


    def begin_training(self, x=None, y=None, minibatch=0):
        updates = self.update_parameters()

        if x is not None and y is not None and minibatch > 1:
            # create shared variables for x and y
            index = create_theano_tensor(self.name + "_index", 0, int)

            train_set_x = theano.shared(x, name=self.name + "_set_x",
                                        allow_downcast=True, borrow=True)

            train_set_y = theano.shared(y, name=self.name + "_set_y",
                                        allow_downcast=True, borrow=True)

            self.train_function = theano.function(inputs=[index],
                                                  outputs=self.cost, updates=updates,
                                                  allow_input_downcast=True,
                                                  givens={
                                                      self.X: train_set_x[index * minibatch: (index + 1) * minibatch],
                                                      self.Y: train_set_y[index * minibatch: (index + 1) * minibatch]
                                                  })

        elif minibatch <=1 and x is None and y is None:
            self.train_function = theano.function(inputs=[self.X, self.Y],
                                                  outputs=self.cost, updates=updates,
                                                  allow_input_downcast=True)
        else:
            self.logger.error(
                "Cannot begin train with the parameters: {0}, {1}, {2}".format(type(x), type(y), minibatch))