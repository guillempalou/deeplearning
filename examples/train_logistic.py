import logging
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger("GENERAL")


sys.path.extend(['/Users/guillem/developer/kaggle/deep_learning'])

from deep_learning.nets.logistic import Logistic
from deep_learning.training.gradient_descent_gpu import gradient_descent
from deep_learning.training.learning_parameters import DescentParameters, ValidationParameters

N = 1000
K = 1
feats = 2
classes = 2

# np.random.seed(seed=0)

points = np.zeros(((N * K * classes), feats + 1))
for c in range(K * classes):
    mean = np.random.random(2)*5
    cov = (0.7 * np.eye(2, 2) + 0.3 * np.random.random((2, 2))) * 0.4
    cov = np.dot(cov, cov.T)  # make it positive semidefinite
    points[c * N:(c + 1) * N, :-1] = np.random.multivariate_normal(mean, cov, N)
    points[c * N:(c + 1) * N, -1] = c % classes

permutation = np.random.permutation(N * K * classes)
X, Y = points[permutation, :-1].astype(np.float32), \
       points[permutation, -1].astype(np.int32)

""" Model definition and training """

model = Logistic("Logistic", feats, classes)

learning_parameters = DescentParameters(epochs=1000, minibatch=100, momentum=0.9)
validation_parameters = ValidationParameters()

logger.info("Begin")
for tc, vc in gradient_descent(model, X, Y, learning_parameters, validation_parameters):
    pass


""" Drawing stuff """

xmin = np.min(points[:, 0]) - 0.1
xmax = np.max(points[:, 0]) + 0.1
ymin = np.min(points[:, 1]) - 0.1
ymax = np.max(points[:, 1]) + 0.1

h = 0.02
xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                     np.arange(ymin, ymax, h))

Z = np.zeros(np.prod(xx.shape))
predict = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = np.argmax(predict, axis=1)
Z = np.reshape(Z, xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, interpolation='nearest')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()
