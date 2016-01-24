import logging
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from data_generation.gaussian_mixtures import generate_random_clouds
from deep_learning.nets.logistic import LogisticNet
from deep_learning.training.gradient_descent import StochasticGradientDescent
from deep_learning.training.losses import cross_entropy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

theano.config.exception_verbosity='high'
theano.config.compute_test_value='off'

learning_parameters = {
    "minibatch": 10,
    "max_iter": 100,
    "learning_rate": 0.1,
}

n_classes = 3
logistic = LogisticNet("logistic", 2, n_classes, activation=T.tanh)

sgd = StochasticGradientDescent(loss=cross_entropy,
                                learning_parameters=learning_parameters)

np.random.seed(0)
X, Y = generate_random_clouds(1000, n_classes, 2, 1)

sgd.fit(logistic, X, Y.astype(np.int32))

""" Drawing stuff """

xmin = np.min(X[:, 0]) - 0.5
xmax = np.max(X[:, 0]) + 0.5
ymin = np.min(X[:, 1]) - 0.5
ymax = np.max(X[:, 1]) + 0.5

h = 0.02
xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                     np.arange(ymin, ymax, h), )

predict = logistic.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = np.argmax(predict, axis=1)
Z = np.reshape(Z, xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, interpolation='nearest')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()