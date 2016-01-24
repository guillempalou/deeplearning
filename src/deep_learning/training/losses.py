import theano
import theano.tensor as T

"""
A bunch of loss definitions for ease
"""

# zero one loss
zero_one_loss = lambda x, y: T.sum(T.neq(T.argmax(x), y))

# log loss or cross entropy error
cross_entropy = lambda x, y: -T.mean(T.log2(x[T.arange(0, y.shape[0]), y]))

# mean squarred error
mse = lambda x, y: T.mean(T.square(x - y))
