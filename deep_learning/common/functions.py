import theano
import theano.tensor as T

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


