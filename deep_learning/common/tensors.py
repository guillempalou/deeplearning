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

    if type(init_method) != list:
        # generate random numbers in the linear activation region for neurons
        ulimit = sqrt(3.0/n_elements)
        llimit = -ulimit
        logger.debug("Initializing tensor of shape {0} with  uniform({1},{2})".format(shape, llimit, ulimit))
        value = rng.uniform(llimit, ulimit, shape)
    elif type(init_method) == list:
        pdf = init_method[0]
        # add more pdfs
        if pdf == 'normal':
            mean = float(init_method[1])
            std = float(init_method[2])
            logger.debug("Initializing tensor of shape {0} with  normal({1},{2})".format(shape, mean, std))
            value = rng.normal(mean, std, shape)

        if pdf == 'uniform':
            llimit = float(init_method[1])
            ulimit = float(init_method[2])
            logger.debug("Initializing tensor of shape {0} with  uniform({1},{2})".format(shape, llimit, ulimit))

            value = rng.uniform(llimit, ulimit, shape)
    elif isinstance(init_method, (int, long, float, complex)):
        logger.debug("Initializing tensor of shape {0} with constant {1}".format(shape, init_method))
        value = init_method * np.ones(shape)


    return theano.shared(value=value.astype(np.float32), name=name)

