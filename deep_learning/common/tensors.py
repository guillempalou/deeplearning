import numpy as np
from numpy.ma import sqrt
import theano
import theano.tensor as T

theano_types = {
    int: [T.iscalar, T.ivector, T.imatrix, T.itensor3, T.itensor4],
    float: [T.fscalar, T.fvector, T.fmatrix, T.ftensor3, T.ftensor4],
}


def create_theano_tensor(name, dims, out_type):
    return theano_types[out_type][dims](name)


def create_shared_variable(name, shape, generator, **kwargs):

    ulimit = 1

    if generator == 'tanh':
        fan_in = kwargs['fan_in']
        fan_out = kwargs['fan_out']
        # generate random numbers in the linear activation region for neurons
        ulimit = sqrt(6.0/np.sum(fan_in + fan_out))

    if generator == 'softmax' or generator == 'logistic':
        fan_in = kwargs['fan_in']
        fan_out = kwargs['fan_out']
        ulimit = 4*sqrt(6.0/np.sum(fan_in + fan_out))

    llimit = -ulimit

    if type(generator) == int or type(generator) == float:
        return theano.shared(value=generator*np.ones(shape).astype(np.float32), name=name)

    value = llimit + np.random.random(shape) * (ulimit - llimit)
    return theano.shared(value=value.astype(np.float32), name=name)

