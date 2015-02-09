import logging
from deep_learning.io.net_io import save_net


class GradientDescent:
    logger = logging.getLogger("GradientDescent")

    def __init__(self, learning_parameters, validation_parameters):
        self.learning_parameters = learning_parameters
        self.validation_parameters = validation_parameters

        # initialize parameters for learning
        self.best_params = None
        self.best_cost = 1e100
        self.iteration = 0
        self.n_batches = 0

        # variables for training. call setup() before doing any training
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def setup(self, model, x=None, y=None):

        self.logger.info("Beginning training for model {0}".format(model.name))
        self.logger.debug("Learning parameters {0}".format(self.learning_parameters))
        self.logger.debug("Validation parameters {0}".format(self.validation_parameters))

        if x is None and y is None:
            model.begin_training(self.learning_parameters, None, None)
        else:
            model.begin_training(self.learning_parameters, x=x, y=y)

    def do_epoch(self, model, x=None, y=None):

        train_cost = 0
        val_cost = 0
        minibatch = self.learning_parameters.minibatch*self.learning_parameters.augmented

        self.n_batches = x.shape[0] // minibatch

        for batch in range(self.n_batches):
            self.logger.debug("Doing batch {0} / {1}".format(batch+1, self.n_batches))
            if x is None and y is None:
                tcost, vcost = model.train(index=batch)
            else:
                start = batch * minibatch
                end = min(x.shape[0], (batch + 1) * minibatch)
                xbatch, ybatch = x[start:end], y[start:end]
                tcost, vcost = model.train(xbatch, ybatch)

            val_cost += vcost
            train_cost += tcost
            self.iteration += 1

        val_cost /= self.n_batches
        train_cost /= self.n_batches

        return train_cost, val_cost

    def train(self, model, data_in_gpu, augmentation, save_file=None, x_train=None, y_train=None, x_val=None, y_val=None):


        self.logger.info("There are a total of {0} batches per epoch".format(self.n_batches))

        do_cross_val = self.validation_parameters is not None and x_val is not None and y_val is not None

        if data_in_gpu:
            if x_train is None or y_train is None:
                self.logger("To train in GPU you need to supply the data to transfer")
                return None
            self.logger.info("Setting up GPU variables for training")
            self.setup(model, x_train, y_train)
        else:
            self.logger.info("Warning: variables will be transferred each time to the GPU")
            self.setup(model)

        self.iteration = 0
        tries = 0
        cost = 1e100

        self.logger.info("Beginning epochs")
        for epoch in range(self.learning_parameters.epochs):
            self.logger.info("Training on Epoch {0}".format(epoch+1))

            x_augmented, y_augmented = augmentation.augment(x_train, y_train)

            train_cost, val_cost = self.do_epoch(model, x_augmented, y_augmented)
            self.logger.info("Mean train error {0}".format(val_cost))

            if do_cross_val:
                x_val_augmented, y_val_augmented = augmentation.augment(x_val, y_val)
                current_cost = self.validate(model, x=x_val_augmented, y=y_val_augmented)
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_params = model.get_parameters_values()
                    if save_file is not None:
                        save_net(save_file, model, self.best_params)

                rel_improvement = (cost - current_cost) / cost
                cost = current_cost

                self.logger.info("Mean test error {0}".format(current_cost))
                self.logger.debug("Best cost to the moment: {0}".format(self.best_cost))

                if rel_improvement < self.learning_parameters.tolerance:
                    tries += 1
                else:
                    tries = 0

                if tries == self.learning_parameters.tries:
                    self.logger.info("Training finished")
                    break

        if not do_cross_val:
            self.best_cost = cost
            self.best_params = model.get_parameters_values()

        return self.best_cost, self.best_params


    def validate(self, model, x=None, y=None, index=None):
        if x is None and y is None:
            return model.test(index=index)
        else:
            n = x.shape[0]
            minibatch = self.learning_parameters.minibatch*self.learning_parameters.augmented
            n_batches = n // minibatch
            tcost = 0
            for batch in range(n_batches):
                start = batch * minibatch
                end = min(x.shape[0], (batch + 1) * minibatch)
                xbatch, ybatch = x[start:end], y[start:end]
                tcost += model.test(xbatch, ybatch)
            return tcost / n_batches


