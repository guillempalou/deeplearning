import logging

from deep_learning.units.activation.dropout_activation import DropOutActivation


class ReLuActivation(DropOutActivation):
    logger = logging.getLogger(__name__ + "." + "ReLuActivation")

    def __init__(self):
        super(ReLuActivation, self).__init__()
        pass