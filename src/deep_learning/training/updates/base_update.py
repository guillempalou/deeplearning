import logging


class BaseUpdate(object):
    logger = logging.getLogger(__name__ + "." + "BaseUpdate")

    def __init__(self, **kwargs):
        super(BaseUpdate, self).__init__()
        pass

    def update_step(self, **kwargs):
        raise NotImplementedError("You should implement the update equation")

    def update_parameters(self, **kwargs):
        """
        This functions is called after each iteration and is ment to update parameters
        for the learning
        :param kwargs:
        :return empty list, no updates
        """
        return []