import logging
import theano
import theano.tensor as T
from deep_learning.common.cost_functions import cross_entropy, mse, l1_norm, l2_sqnorm, accuracy


class CostController:
    logger = logging.getLogger("CostController")

    def __init__(self, model, learning_parameters):
        self.train_loss = learning_parameters.train_loss
        self.test_loss = learning_parameters.test_loss
        self.l1 = learning_parameters.l1
        self.l2 = learning_parameters.l2
        self.augmented = learning_parameters.augmented
        self.minibatch = learning_parameters.minibatch

        # theano functions
        self.train_function_raw = None
        self.test_function_raw = None
        self.train_cost = None
        self.test_cost = None

        # get model parameters
        self.X = model.X
        self.Y = model.Y
        self.model_params = model.params


    def setup_costs(self, model):
        # train with minibatch. we should transfer to the GPU for speed
        self.logger.debug("Compiling train cost function")
        self.train_cost = self.train_cost_function(model)
        self.logger.debug("Compiling test cost function")
        self.test_cost = self.test_cost_function(model)
        # return the train for updates
        return self.train_cost

    def setup_updates(self, updates):
        self.setup_train_functions(updates)
        self.setup_test_functions()


    def train(self, x, y):
        return self.train_function_raw(x, y)

    def test(self, x, y):
        return self.test_function_raw(x, y)

    def train_cost_function(self, model):
        # TODO add more costs functions
        loss = None

        if self.train_loss == 'crossentropy':
            loss = cross_entropy(model.transform(self.X), self.Y)

        if self.train_loss == 'mse':
            loss = mse(model.transform(self.X), self.Y)

        # l1 and l2 only affect to the weights of the last layer
        # by model construction they lie always on the -2 position
        w_last = self.model_params[-2]

        return loss + self.l1 * l1_norm([w_last]) + self.l2 * l2_sqnorm([w_last])

    def test_cost_function(self, model):

        xout, yout = self.deaugment(model)

        z = None
        if self.test_loss == 'accuracy':
            z = accuracy(T.argmax(xout, axis=1), yout)

        if self.test_loss == 'crossentropy':
            z = cross_entropy(xout, yout)

        return z

    def deaugment(self, model):
        t = model.transform(self.X, 'test')
        yout = self.Y

        if self.augmented > 1:
            n = self.minibatch
            dups = self.augmented
            tr = t.reshape((n, dups, model.output_shape))
            xout = tr.mean(axis=1)
            # todo inefficient use of memory and cpu
            yout = self.Y[::dups]
        else:
            xout = t

        return xout, yout

    def setup_train_functions(self, updates):
        self.logger.debug("Compiling updates function")
        self.logger.debug("Setting up functions for raw inputs")
        self.train_function_raw = theano.function(inputs=[self.X, self.Y],
                                                  outputs=[self.train_cost, self.test_cost],
                                                  updates=updates,
                                                  allow_input_downcast=True)


    def setup_test_functions(self):
        self.logger.debug("Setting up test function for raw inputs")
        self.test_function_raw = theano.function(inputs=[self.X, self.Y],
                                                 outputs=self.test_cost,
                                                 allow_input_downcast=True)





