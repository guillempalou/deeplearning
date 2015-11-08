import logging

from deep_learning.units.activation.dropout_activation import DropOutActivation


class TanhActivation(DropOutActivation):
    logger = logging.getLogger(__name__ + "." + "TanhActivation")

    def __init__(self):
        super(TanhActivation, self).__init__()
        pass