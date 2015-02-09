import logging
import numpy as np
from numpy.ma import sqrt
import theano
import theano.tensor as T

logger = logging.getLogger("Tensors")

theano_types = {
    int: [T.iscalar, T.ivector, T.imatrix, T.itensor3, T.itensor4],
    float: [T.fscalar, T.fvector, T.fmatrix, T.ftensor3, T.ftensor4],
}


def create_theano_tensor(name, dims, out_type):
    return theano_types[out_type][dims](name)


def create_shared_variable(name, shape, init_method):

    value = None
    rng = np.random.RandomState()

    n_elements = np.prod(shape)

    logger.debug("Creating tensor {0} with shape {1} with initialization {2}".format(name, shape, init_method))

    if init_method == 'tanh':
        # generate random numbers in the linear activation region for neurons
        ulimit = sqrt(3.0/n_elements)
        llimit = -ulimit
        value = rng.uniform(llimit, ulimit, shape)

    if init_method == 'softmax' or init_method == 'logistic':
        ulimit = sqrt(3.0/n_elements)
        llimit = -ulimit
        value = rng.uniform(llimit, ulimit, shape)

    if init_method == 'relu':
        value = rng.normal(0, np.sqrt(1.0/n_elements), shape)

    if type(init_method) == list:
        pdf = init_method[0]
        # add more pdfs
        if pdf == 'normal':
            mean = float(init_method[1])
            std = float(init_method[2])
            value = rng.normal(mean, std, shape)

        if pdf == 'uniform':
            llimit = float(init_method[1])
            ulimit = float(init_method[2])
            value = rng.uniform(llimit, ulimit, shape)
    elif isinstance(init_method, (int, long, float, complex)):
        value = init_method * np.ones(shape)


    return theano.shared(value=value.astype(np.float32), name=name)

