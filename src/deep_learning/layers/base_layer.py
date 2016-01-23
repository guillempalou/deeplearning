class BaseLayer(object):

    def __init__(self, **kwargs):
        # set some basic variables that all inputs need to have
        self.name = kwargs["name"]
        self.in_shape = kwargs["in_shape"]
        self.out_shape = kwargs["out_shape"]

    def transform(self, x, **kwargs):
        """
        Constructs the theano function for this layer
        :param x: theano variable
        :param kwargs:
        """
        raise NotImplementedError("No method defined")

    def input_variable(self):
        """
        returns theano variable type
        """
        raise NotImplementedError("Each layer should define the input")

    def output_variable(self):
        """
        returns theano variable type
        """
        raise NotImplementedError("Each layer should define the output")

    def get_parameters(self):
        """
        Get a dictionary with the shared variables composing the layer
        """
        raise NotImplementedError("You should implement get_parameters()")

    def __str__(self):
        return "Layer {0} - ({1},{2})".format(self.name, self.in_shape, self.out_shape)
