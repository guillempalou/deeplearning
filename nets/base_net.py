import abc
import theano
import theano.tensor as T


class BaseNet(object):

    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def cost_function(self, x, y):
        pass

    @abc.abstractmethod
    def gradient(self, layer=None):
        pass

    @abc.abstractmethod
    def begin_training(self):
        pass

    @abc.abstractmethod
    def update_parameters(self):
        pass

    @abc.abstractmethod
    def train(self, x, y):
        pass

    @abc.abstractmethod
    def transform(self, x):
        pass

    # TODO define abstract property for prediction
    # TODO define abstract property for layers

    def get_parameters(self):
        return self.params
