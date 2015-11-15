import logging

from deep_learning.units.activation.dropout_activation import DropOutActivation


class ReLuActivation(DropOutActivation):
    """
    Implements rectified linear units with dropout
    """
    logger = logging.getLogger(__name__ + "." + "ReLuActivation")

    def __init__(self, **kwargs):
        super(ReLuActivation, self).__init__(**kwargs)

    def __call__(self, x, **kwargs):
        x_dropped = super(ReLuActivation, self).__call__(x, **kwargs)
        return x_dropped * (x_dropped > 0)
