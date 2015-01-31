import logging



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
        minibatch = self.learning_parameters.minibatch


        for batch in range(self.n_batches):
            if x is None and y is None:
                tcost, vcost = model.train(index=batch)
            else:
                start = batch * minibatch
                end = min(x.shape[0] - 1, (batch + 1) * minibatch)
                xbatch, ybatch = x[start:end], y[start:end]
                tcost, vcost = model.train(xbatch, ybatch)

            val_cost += vcost
            train_cost += tcost
            self.iteration += 1

        val_cost /= self.n_batches
        train_cost /= self.n_batches

        return train_cost, val_cost

    def train(self, model, data_in_gpu=True, x_train=None, y_train=None, x_val=None, y_val=None):

        self.n_batches = x_train.shape[0] // self.learning_parameters.minibatch
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

            train_cost, val_cost = self.do_epoch(model, x_train, y_train)

            if do_cross_val:
                current_cost = model.test(x=x_val, y=y_val)
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_params = model.get_parameters_values()

                rel_improvement = (cost - current_cost) / cost
                cost = current_cost

                self.logger.info("Epoch {0} - {1} mean train error and {2} mean test error".format(epoch,
                                                                                                   val_cost,
                                                                                                   current_cost))
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
            return model.test(x=x, y=y)


