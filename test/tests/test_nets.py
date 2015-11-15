from deep_learning.nets.logistic import LogisticNet
import theano
import theano.tensor as T

def test_logistic():
    net = LogisticNet("logistic", 2, 2)
    x = T.fvector('x')
    f = theano.function([x], net.transform(x))
    print(f([1, 2]))