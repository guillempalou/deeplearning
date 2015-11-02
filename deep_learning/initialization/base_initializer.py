import logging


class BaseInitializer(object):
    logger = logging.getLogger("BaseInitializer")

    def __init__(self):
        self.logger.debug("Instantiating initializer object")

    def __call__(self, **kwargs):
        return NotImplementedError("You should subclass the method")
