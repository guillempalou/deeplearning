import json
import pickle
from deep_learning.nets.net_factory import create_net_from_file, create_net_from_dict


def save_net(filename, nnet, parameters=None, protocol=2):
    f = open(filename, "wb")
    pickle.dump(nnet.name, f, protocol=protocol)
    pickle.dump(nnet.definition(), f, protocol=protocol)
    if parameters is None:
        params = nnet.params
    else:
        params = parameters

    for param in params:
        pickle.dump(param, f, protocol=protocol)
    f.close()


def load_net(filename):
    f = open(filename, "rb")
    name = pickle.load(f)
    definition = pickle.load(f)
    definition[0]['filters'] = [20, 5, 5]
    definition[1]['filters'] = [50, 5, 5]
    nnet = create_net_from_dict(name, definition)
    for param in nnet.params:
        v = pickle.load(f)
        param.set_value(v)
    return nnet