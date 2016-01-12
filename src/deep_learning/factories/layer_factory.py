from deep_learning.factories.initializer_factory import create_initializer
from deep_learning.layers.softmax_layer import SoftMaxLayer


def create_softmax_layer(name, input_units, output_units, initializer):
    """
    Creates a softmax layer with a given input and output shapes
    :param name: name of the layer
    :param input_units: number
    :param output_units: number
    :param initializer: initializer dictionary
    :return:
    """
    assert isinstance(input_units, int), "Input units needs to be an int"
    assert isinstance(output_units, int), "Output units needs to be an int"
    assert input_units >= 1, "Input units needs to be positive"
    assert output_units >= 1, "Output units needs to be positive"

    initializer = {
        "w": create_initializer(initializer, (input_units, output_units)),
        "b": create_initializer(initializer, output_units)
    }

    return SoftMaxLayer(in_shape=input_units, out_shape=output_units, initializer=initializer)

def create_hidden_layer():
    pass

def create_convolutional_layer():
    pass