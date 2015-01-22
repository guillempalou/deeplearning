import abc
import numpy as np
import theano
import theano.tensor as T
from deep_learning.common.tensors import create_theano_tensor


class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, n_in, n_out, in_type=float, truth_type=float):
        self.name = name

        # take into account the minibatch
        self.X = create_theano_tensor(name + "_X", n_in.ndim + 1, in_type)
        self.Y = create_theano_tensor(name + "_Y", n_out.ndim, truth_type)
        self.n_in = n_in
        self.n_out = n_out
        self.type_in = in_type
        self.type_truth = truth_type

        # define the params on the sublayers

        # define the function to predict
        X_predict = create_theano_tensor(name + "X_test", n_in.ndim + 1, in_type)
        self.predict = theano.function(inputs=[X_predict],
                                       outputs=self.transform(X_predict),
                                       allow_input_downcast=True)

    @abc.abstractmethod
    def transform(self, x):
        return "No method defined"

    @abc.abstractmethod
    def get_parameters(self):
        return self.params

    @abc.abstractmethod
    def get_weights(self):
        return self.w.get_value()

    @abc.abstractmethod
    def get_bias(self):
        return self.b.get_value()

    def l1_norm(self):
        c = 0
        for p in self.params:
            c += T.sum(abs(p))
        return c

    def l2_sqnorm(self):
        sq = 0
        for p in self.params:
            sq += T.sum(T.square(p))
        return sq

