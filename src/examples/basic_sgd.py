import logging
import numpy as np

import theano
import theano.tensor as T

from data_generation.gaussian_mixtures import generate_random_clouds
from deep_learning.nets.logistic import LogisticNet
from deep_learning.nets.mlp import MLP
from deep_learning.training.gradient_descent import StochasticGradientDescent
from deep_learning.training.updates.sgd_update import SGDUpdate

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

learning_parameters = {
    "minibatch": 10,
    "max_iter": 100
}


mlp = MLP("mlp", 2, 2, 2)
loss = lambda x, y: -T.mean(T.log2(x[:, y]))

update = SGDUpdate(loss=loss, model=mlp)
sgd = StochasticGradientDescent(loss=loss,
                                learning_parameters=learning_parameters,
                                updates=update)

X, Y = generate_random_clouds(1000, 2, 2)

sgd.fit(mlp, X, Y.astype(np.int32))
