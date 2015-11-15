import logging


class BaseInitializer(object):
    logger = logging.getLogger("BaseInitializer")

    def __init__(self, name, shape):
        self.logger.debug("Instantiating initializer object {0} - {1})".format(name, shape))
        self.name = name
        self.shape = shape

    def create_shared(self, **kwargs):
        """
        Abstract method to create a theano shared variable
        :param kwargs:
        :return:
        """
        return NotImplementedError("You should subclass the method")
