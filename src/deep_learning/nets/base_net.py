import abc
import logging

import theano

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
                                       outputs=self.transform(self.X, 'test'),
                                       allow_input_downcast=True)


    @abc.abstractmethod
    def transform(self, x, mode='train'):
        pass

    def get_parameters_values(self):
        return [p.get_value() for p in self.params]


    def set_paramaters_values(self, values):
        for v, p in zip(values, self.params):
            p.set_value(v)


    def definition(self):
        return [layer.definition() for layer in self.layers]


    def __str__(self):
        return self.definition()