import logging
import theano
import theano.tensor as T
from deep_learning.training.updates.base_update import BaseUpdate


class SGDUpdate(BaseUpdate):
    """
    Class implementing a simple gradient update
    Optionally, the learning rate can be changed from one epoch to another
    """
    logger = logging.getLogger(__name__ + "." + "SGDUpdate")

    def __init__(self, **kwargs):
        super(SGDUpdate, self).__init__(**kwargs)
        self.learning_rate = kwargs.get("learning_rate", None)

    def update_step(self, model, **kwargs):
        pass

    def update_parameters(self, **kwargs):
        """
        Updates the model parameter
        :param kwargs: all parameters needed
        :return: list of update equations
        """
        model = kwargs["model"]
        loss = kwargs["loss"]
        model_parameters = model.get_parameters()
        gradient = T.grad(loss, model_parameters)

        return [(w, w - self.learning_rate * gradient[i])
                for i, w in enumerate(model_parameters)]
