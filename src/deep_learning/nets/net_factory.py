import logging

import yaml
import numpy as np

from src.deep_learning import ConvolutionalLayer
from src.deep_learning import HiddenLayer
from src.deep_learning import SoftMaxLayer
from src.deep_learning import DeepNet

logger = logging.getLogger("NET_FACTORY")


def create_net_from_file(name, file_path):
    input_file = open(file_path)
    definition = yaml.load(input_file)
    input_file.close()
    return create_net_from_dict(name, definition)

def create_net_from_dict(name, definition):
    previous_output = None
    layers = []
    for layer_def in definition:
        layer = create_layer(previous_output, layer_def)

        if layer is None:
            return None

        previous_output = layer.output_shape
        layers.append(layer)

    return DeepNet(name, layers)


def create_layer(previous_output, layer_definition):
    if previous_output is None and 'input_shape' not in layer_definition:
        logger.error("The input shape should be specified at the first layer")
        return None

    # must have attributes
    name = layer_definition['name']
    layer_type = layer_definition['type']
    input_shape = previous_output if previous_output is not None else layer_definition['input_shape']
    activation = 'tanh' if 'activation' not in layer_definition else layer_definition['activation']
    dropout = 0 if 'dropout' not in layer_definition else layer_definition['dropout']
    init_weights = None if 'init_weights' not in layer_definition else layer_definition['init_weights']
    init_bias = None if 'init_bias' not in layer_definition else layer_definition['init_bias']

    layer = None

    if layer_type == 'convolutional':
        filters = layer_definition['filters']
        stride = (1, 1) if 'stride' not in layer_definition else layer_definition['stride']
        pool = (1, 1) if 'pool' not in layer_definition else layer_definition['pool']

        layer = ConvolutionalLayer(name, input=input_shape, filters=filters, stride=stride, pool=pool,
                                   activation=activation, dropout=dropout, init_bias=init_bias, init_weights=init_weights)

    if layer_type == 'hidden':
        n_input = np.prod(input_shape) if type(input_shape) != int else input_shape
        n_output = layer_definition['output_shape']
        layer = HiddenLayer(name, n_in=n_input, n_out=n_output, activation=activation, dropout=dropout,
                            init_weights=init_weights,
                            init_bias=init_bias)

    if layer_type == 'softmax':
        n_input = np.prod(input_shape) if type(input_shape) != int else input_shape
        n_output = layer_definition['output_shape']
        layer = SoftMaxLayer(name,
                             n_in=n_input,
                             n_out=n_output,
                             init_weights=init_weights,
                             init_bias=init_bias)

    return layer