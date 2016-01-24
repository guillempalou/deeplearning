import logging
import numpy as np
import theano
import theano.tensor as T

from deep_learning.theano_utils import get_symbolic_variable
from deep_learning.training.updates.vanilla_update import VanillaUpdate


class StochasticGradientDescent(object):
    """
    Class controling the fitting of a neural net
    using gradient descent
    """
    logger = logging.getLogger("StochasticGradientDescent")

    def __init__(self,
                 loss,
                 learning_parameters,
                 updates=VanillaUpdate,
                 callback=None):

        # initialize parameters for learning
        self.best_loss = 1e100

        # variables for learning
        self.loss = loss
        self.learning_parameters = learning_parameters
        self.updates = updates(**learning_parameters)

        self.callback = callback
        self.iteration = 0
        self.n_batches = 0

        # functions for theano
        self.train = None
        self.test = None

    def fit(self, model, X, Y):

        # get input and output theano variables based on X and Y
        tX = get_symbolic_variable(X, "X")
        tY = get_symbolic_variable(Y, "Y")

        if self.iteration == 0:
            self.setup(model, tX, tY)

        for self.iteration in range(self.learning_parameters["max_iter"]):

            # self do a full iteration over the data
            self.do_epoch(model, X, Y)

            # update the learning parameters
            self.updates.update_step(model)

            # if the user has specified a call back call it
            # with all the possible information
            if self.callback is not None:
                self.callback(self)

    def setup(self, model, X, Y):

        self.logger.info("Beginning training for model {0}".format(model.name))
        self.logger.debug("Learning parameters {0}".format(self.learning_parameters))

        # initialize the train function
        # TODO add possible regularization on parameters
        loss_function = self.loss(model.transform(X, mode="train"), Y)

        # get the update equations for the model parameters
        # using the train function
        updates = self.updates.update_parameters(model=model, loss=loss_function)

        model.setup(X, Y, loss_function, updates)

    def do_epoch(self, model, X, Y):
        """
        Do a full pass to the data while training the parameters
        """
        minibatch = self.learning_parameters["minibatch"]
        n_samples = X.shape[0]
        n_minibaches = np.ceil(n_samples / minibatch).astype(int)

        start = 0
        epoch_loss = 0
        for bach in range(n_minibaches):
            end = min(n_samples, start + minibatch)
            x_batch = X[start:end, ...]
            y_batch = Y[start:end, ...]

            # compute the loss and update
            loss = model.fit(x_batch, y_batch)
            self.logger.debug("Loss on train minibatch {0}: ".format(loss))

            epoch_loss += loss
            start += minibatch

        self.logger.info("Epoch {0} loss {1}".format(self.iteration, epoch_loss))