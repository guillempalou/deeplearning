import logging

from deep_learning.units.activation.base_activation import BaseActivation


class DropOutActivation(BaseActivation):
    logger = logging.getLogger(__name__ + "." + "DropOutActivation")

    def __init__(self):
        super(DropOutActivation, self).__init__()
        pass