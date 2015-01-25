import sys
import pickle

sys.path.extend(['/Users/guillem/developer/kaggle/deep_learning'])

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from deep_learning.nets.logistic import Logistic

# from deep_learning.nets.mlp import MLP
from deep_learning.nets.mlp import MLP

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

xmin = np.min(points[:, 0]) - 0.1
xmax = np.max(points[:, 0]) + 0.1
ymin = np.min(points[:, 1]) - 0.1
ymax = np.max(points[:, 1]) + 0.1

h = 0.02
xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                     np.arange(ymin, ymax, h))

permutation = np.random.permutation(N * K * classes)
X, Y = points[permutation, :-1].astype(np.float32), \
       points[permutation, -1].astype(np.int32)

model = Logistic("binary", feats, classes, learning_rate=0.001, l1=0.001, l2=0.001)

minibatch = 1000
if minibatch <= 1:
    model.begin_training()
else:
    model.begin_training(x=X, y=Y, minibatch=minibatch)

passes = 10000

max_tries = 3
tolerance = 1e-5
ant_cost = 1e100

tries = 0

for epoch in range(passes):
    cost = 0

    if minibatch > 1:
        n_batches = N * K * classes // minibatch
        for i in range(n_batches):
            # comput the cost
            c = model.train_function(i)
            cost += c / n_batches
    else:
        for i in range(N * K * classes):
            # TODO we should be able to specify a single output
            c = model.train_function(X[i:i + 1, :], Y[i:i + 1])
            cost += c / (N*K*classes)

    print("Mean cost for epoch {0} is {1}".format(epoch + 1, cost))
    print("Computed {0} examples".format((epoch + 1) * N * K * classes))

    cost_variation = (ant_cost - cost) / ant_cost
    print("Cost relative variation {0}".format(cost_variation))

    if cost_variation < tolerance:
        tries += 1
    else:
        tries = 0

    ant_cost = cost

    if tries == max_tries:
        print("Learning complete")
        break


Z = np.zeros(np.prod(xx.shape))
predict = model.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = np.argmax(predict, axis=1)
Z = np.reshape(Z, xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, interpolation='nearest')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.show()
