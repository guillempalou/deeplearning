import logging
import theano
import theano.tensor as T
import numpy as np

class GradientController:

    logger = logging.getLogger("GradientController")

    def __init__(self, learning_parameters):

        self.learning_parameters = learning_parameters

        # setup initial learning rate, decay and momentum
        self.e0 = self.learning_parameters.learning_rate
        self.tau = self.learning_parameters.start_decay
        self.mu = self.learning_parameters.momentum

        self.update_rates = None
        self.learning_rate = None
        self.epochs = None

    def setup_updates(self, cost, model, epoch=0):
        self.logger.debug("Setup updates function")
        # shared variables to control the progress
        self.epochs = theano.shared(np.float32(epoch), "epoch", allow_downcast=True)
        self.learning_rate = theano.shared(np.float32(self.learning_parameters.learning_rate),
                                           "learning_rate",
                                           allow_downcast=True)
        self.update_rates = self.update_rates_function()
        return self.update_parameters(cost, model.params)

    def custom_max(self, a, b):
        return a * T.ge(a, b) + b * T.le(a, b)

    def update_rates_function(self):
        epoch = T.fscalar("epochs")
        return theano.function(inputs=[epoch],
                               outputs=[self.epochs, self.learning_rate],
                               updates=[[self.epochs, epoch],
                                        [self.learning_rate, self.e0 * self.tau / self.custom_max(epoch, self.tau)]])


    def gradient(self, cost, model_parameters):
        return T.grad(cost=cost, wrt=model_parameters)


    def update_parameters(self, cost, model_parameters):
        grad = self.gradient(cost, model_parameters)

        updates = []

        for i, param in enumerate(model_parameters):
            init = np.zeros(param.get_value().shape, dtype=np.float32)
            param_update = theano.shared(init,
                                         name=param.name + "_update",
                                         broadcastable=param.broadcastable, borrow=True)

            updates.append([param_update, self.mu * param_update + (1. - self.mu) * grad[i]])
            updates.append([param, param - self.learning_rate * param_update])

        return updates
