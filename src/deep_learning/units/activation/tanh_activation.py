import logging
import theano
import theano.tensor as T

from deep_learning.units.activation.dropout_activation import DropOutActivation


class TanhActivation(DropOutActivation):
    """
    Implements Tanh activation function with dropout
    """
    logger = logging.getLogger(__name__ + "." + "TanhActivation")

    def __init__(self, **kwargs):
        super(TanhActivation, self).__init__(**kwargs)
        pass

    def __call__(self, x, **kwargs):
        x_dropped = super(TanhActivation, self).__call__(x, **kwargs)
        return T.tanh(x_dropped)
