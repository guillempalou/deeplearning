import theano
import theano.tensor as T


def get_symbolic_variable(X, name):
    """
    Returns a type of simbolic variable based on a numpy array dimensions and type
    :param X: numpy array
    """
    dim = X.ndim
    assert 1 <= dim <= 4, "Dimensions should be between 1 and 4"

    dtype = X.dtype.name
    dtype = "int" if dtype.startswith("int") else dtype
    dtype = "float" if dtype.startswith("float") else dtype

    assert (dtype == "int") or (dtype == "float"), \
        "Type {0} not supported".format(dtype.name)

    types = {
        0: {"int": T.iscalar, "float": T.fscalar},
        1: {"int": T.ivector, "float": T.fvector},
        2: {"int": T.imatrix, "float": T.fmatrix},
        3: {"int": T.tensor3, "float": T.ftensor3},
        4: {"int": T.itensor4, "float": T.ftensor4}
    }

    return types[dim][dtype](name)
