import logging


class BaseActivation(object):
    """
    Provides basic functionality and methods for activation functions
    """
    logger = logging.getLogger(__name__ + "." + "BaseActivation")

    def __init__(self):
        pass

    def __call__(self, x, **kwargs):
        """
        Return the identity activation
        :param x:
        :param kwargs:
        :return:
        """
        return x
