import logging
import theano
import theano.tensor as T
import numpy as np

from deep_learning.units.activation.base_activation import BaseActivation


class DropOutActivation(BaseActivation):
    """
    Class that implements the dropout from Hinton
    Hinton et al. 'Improving neural networks by preventing co-adaptation of feature detectors'
    """
    logger = logging.getLogger(__name__ + "." + "DropOutActivation")

    def __init__(self, **kwargs):
        super(DropOutActivation, self).__init__()
        if "dropout" in kwargs:
            self.dropout = kwargs["dropout"]
        else:
            self.dropout = 0

    def __call__(self, x, **kwargs):
        if kwargs["mode"] == "train":
            return self.drop_output(x) / (1-self.dropout)
        else:
            return x

    def drop_output(self, x):
        """
        Drops output according the dropout probability
        :param x: variable to drop
        :return: x with some entries with 0
        """
        rng = np.random.RandomState()
        srng = T.shared_randomstreams.RandomStreams(rng.randint(100000))
        mask = srng.binomial(n=1, p=1 - self.dropout, size=x.shape)
        return x * T.cast(mask, theano.config.floatX)