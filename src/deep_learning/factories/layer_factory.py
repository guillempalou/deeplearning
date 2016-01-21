import logging

from deep_learning.factories.initializer_factory import create_initializer, InitializerType
from deep_learning.layers.convolutional import Convolutional2DLayer
from deep_learning.layers.hidden_layer import HiddenLayer
from deep_learning.layers.softmax_layer import SoftMaxLayer

logger = logging.getLogger("layer_factory")


# TODO add logging
# TODO all string constant move to proper definitions
# TODO add docstrings

def create_layer_from_dict(layer_definition):
    """
    Creates a layer from a dictionary
    :param layer_definition: dict
    :return: layer
    """
    name = layer_definition["name"]
    input_shape = layer_definition["input"]

    type = layer_definition.get("type", "hidden")

    activation = layer_definition.get("activation", None)

    initializers = layer_definition.get("initialization", {"initializer": InitializerType.FanInOut})

    # create a layer according to the type
    if type == "softmax":
        output_shape = layer_definition["output"]
        return create_softmax_layer(name,
                                    input_shape,
                                    output_shape,
                                    initializers)
    elif type == "hidden":
        output_shape = layer_definition["output"]
        return create_hidden_layer(name,
                                   input_shape,
                                   output_shape,
                                   initializers,
                                   activation=activation)
    elif type == "convolutional":
        n_filters = layer_definition["n_filters"]
        filter_shape = layer_definition["filter"]
        return create_convolutional_2d_layer(name,
                                             input_shape,
                                             n_filters,
                                             filter_shape=filter_shape,
                                             initializer_types=initializers,
                                             activation=activation)
    else:
        # TODO support custom layers
        pass

def create_softmax_layer(name, input_units, output_units, initializer_types):
    """
    Creates a softmax layer with a given input and output shapes
    :param name: name of the layer
    :param input_units: number
    :param output_units: number
    :param initializer_types: initializer dictionary
    :return: softmax layer
    """
    assert isinstance(input_units, int), "Input units needs to be an int"
    assert isinstance(output_units, int), "Output units needs to be an int"
    assert input_units >= 1, "Input units needs to be positive"
    assert output_units >= 1, "Output units needs to be positive"

    initializers = _create_weight_and_bias_inits(name, (input_units, output_units), output_units, initializer_types)

    return SoftMaxLayer(name=name, in_shape=input_units, out_shape=output_units, initializer=initializers)


def create_hidden_layer(name, input_units, output_units, initializer_types, activation):
    """
    Creates a softmax layer with a given input and output shapes
    :param name: name of the layer
    :param input_units: number
    :param output_units: number
    :param initializer_types: initializer dictionary
    :param layer_params: layer additional parameters
    :return: hidden layer
    """
    assert isinstance(input_units, int), "Input units needs to be an int"
    assert isinstance(output_units, int), "Output units needs to be an int"
    assert input_units >= 1, "Input units needs to be positive"
    assert output_units >= 1, "Output units needs to be positive"

    initializers = _create_weight_and_bias_inits(name, (input_units, output_units), output_units, initializer_types)

    return HiddenLayer(name=name,
                       in_shape=input_units, out_shape=output_units,
                       initializer=initializers, activation=activation)


def create_convolutional_2d_layer(name, input_shape, n_filters, filter_shape, initializer_types, **kwargs):
    """
    Creates a 2D convolutional layer given input and filter parameters
    :param name: name of the layer
    :param input_shape: input shape (channels, rows, cols)
    :param n_filters: number of output filters
    :param filter_shape: (rows, cols)
    :param initializer_types: initializer dictionary
    :param kwargs: layer additional arguments (activation, ...)
    :return: convolutional layer
    """
    output_shape = (n_filters,
                    input_shape[1] - filter_shape[0] + 1,
                    input_shape[2] - filter_shape[1] + 1)

    w_shape = (n_filters, input_shape[0]) + tuple(filter_shape)
    b_shape = n_filters

    initializers = _create_weight_and_bias_inits(name, w_shape, b_shape, initializer_types)
    return Convolutional2DLayer(name=name,
                                in_shape=input_shape,
                                out_shape=output_shape,
                                filter=filter_shape,
                                initializer=initializers,
                                **kwargs)


def _create_weight_and_bias_inits(name, w_shape, b_shape, initializer_types):
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
        "w": create_initializer(name + "_w", w_shape, **initializers["w"]),
        "b": create_initializer(name + "_b", b_shape, **initializers["b"])
    }

    return initializer
