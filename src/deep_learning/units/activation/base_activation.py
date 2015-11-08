import logging


class BaseActivation(object):
    """
    Provides basic functionality and methods for activation functions
    """
    logger = logging.getLogger(__name__ + "." + "BaseActivation")

    def __init__(self):
        pass

    def __call__(self, x, **kwargs):
        raise NotImplementedError("You need to implement activation __call__()")