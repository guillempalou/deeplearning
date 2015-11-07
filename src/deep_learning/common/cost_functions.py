import theano
import theano.tensor as T

def mse(x, y):
    return T.mean(T.sum(T.square(x - y)))

def cross_entropy(x, y):
    return -T.mean(T.log(x[T.arange(y.shape[0]), y]))

def accuracy(x, y):
    return T.mean(T.neq(x, y))


# return the l1 norm for the parameters of the net
def l1_norm(params):
    c = 0
    for p in params:
        c += T.sum(abs(p))
    return c

# return the l2 norm for the parameters of the net
def l2_sqnorm(params):
    sq = 0
    for p in params:
        sq += T.sum(p**2)
    return sq