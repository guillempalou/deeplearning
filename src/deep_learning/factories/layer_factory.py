from deep_learning.factories.initializer_factory import create_initializer
from deep_learning.layers.hidden_layer import HiddenLayer
from deep_learning.layers.softmax_layer import SoftMaxLayer


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
    assert len(initializer_types) == 2, "Initializer types should contain two elements"
    assert "w" in initializer_types, "Initializer for weights 'w' not present"
    assert "b" in initializer_types, "Initializer for weights 'b' not present"

    initializer = {
        "w": create_initializer(name + "_w", (input_units, output_units), **initializer_types["w"]),
        "b": create_initializer(name + "_b", output_units, **initializer_types["b"])
    }

    return SoftMaxLayer(name=name, in_shape=input_units, out_shape=output_units, initializer=initializer)

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
    assert len(initializer_types) == 2, "Initializer types should contain two elements"
    assert "w" in initializer_types, "Initializer for weights 'w' not present"
    assert "b" in initializer_types, "Initializer for weights 'b' not present"

    initializer = {
        "w": create_initializer(name + "_w", (input_units, output_units), **initializer_types["w"]),
        "b": create_initializer(name + "_b", output_units, **initializer_types["b"])
    }

    return HiddenLayer(in_shape=input_units, out_shape=output_units, initializer=initializer, activation=activation)

def create_convolutional_layer():
    pass