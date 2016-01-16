import logging

from deep_learning.factories.initializer_factory import create_initializer
from deep_learning.layers.hidden_layer import HiddenLayer
from deep_learning.layers.softmax_layer import SoftMaxLayer

logger = logging.getLogger("layer_factory")

def create_softmax_layer(name, input_units, output_units, initializer_types):
    """
    Creates a softmax layer with a given input and output shapes
    :param name: name of the layer
    :param input_units: number
    :param output_units: number
    :param initializer_types: initializer dictionary
    :return:
    """
    assert isinstance(input_units, int), "Input units needs to be an int"
    assert isinstance(output_units, int), "Output units needs to be an int"
    assert input_units >= 1, "Input units needs to be positive"
    assert output_units >= 1, "Output units needs to be positive"

    initializers = _create_weight_and_bias_inits(name, input_units, output_units, initializer_types)

    return SoftMaxLayer(name=name, in_shape=input_units, out_shape=output_units, initializer=initializers)

def create_hidden_layer(name, input_units, output_units, initializer_types, activation):
    """
    Creates a softmax layer with a given input and output shapes
    :param name: name of the layer
    :param input_units: number
    :param output_units: number
    :param initializer_types: initializer dictionary
    :param layer_params: layer additional parameters
    :return:
    """
    assert isinstance(input_units, int), "Input units needs to be an int"
    assert isinstance(output_units, int), "Output units needs to be an int"
    assert input_units >= 1, "Input units needs to be positive"
    assert output_units >= 1, "Output units needs to be positive"

    initializers = _create_weight_and_bias_inits(name, input_units, output_units, initializer_types)

    return HiddenLayer(name=name,
                       in_shape=input_units, out_shape=output_units,
                       initializer=initializers, activation=activation)

def create_convolutional_layer():
    pass


def _create_weight_and_bias_inits(name, input_units, output_units, initializer_types):

    initializers = {}
    # check if the dictionary contains a key "initializer"
    # if it does, we assume the initialization is the same
    if "initializer" in initializer_types:
        logger.info("Initializer types contains 1 elements, both weights and bias will be initialized equally")
        initializers["w"] = initializer_types
        initializers["b"] = initializer_types
    else:
        assert "w" in initializer_types, "Initializer for weights 'w' not present"
        assert "b" in initializer_types, "Initializer for weights 'b' not present"
        initializers = initializer_types

    initializer = {
        "w": create_initializer(name + "_w", (input_units, output_units), **initializers["w"]),
        "b": create_initializer(name + "_b", output_units, **initializers["b"])
    }

    return initializer