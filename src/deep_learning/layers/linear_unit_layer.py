from theano import tensor as T
from deep_learning.layers.base_layer import BaseLayer
from deep_learning.units.activation.base_activation import BaseActivation


class LinearUnitLayer(BaseLayer):
    """
    Base class definit a feed forward layer of linear units with a (possibly)
    non linear activation function.
    """

    def __init__(self, **kwargs):
        super(LinearUnitLayer, self).__init__(**kwargs)
        self.w = kwargs["initializer"]["w"].create_shared()
        self.b = kwargs["initializer"]["b"].create_shared()

        # our activation function is the identity by default
        self.activation = BaseActivation()

    def transform(self, x, **kwargs):
        """
        Transform the input using softmax
        :param x: input vector
        :param kwargs: additional arguments (ignored)
        :return: vector
        """
        return self.activation(T.dot(x, self.w) + self.b, **kwargs)

    def get_weights(self):
        """
        Returns the weights
        :return: vector w
        """
        return self.w.get_value()

    def get_bias(self):
        """
        Returns the layer bias
        :return: vector b
        """
        return self.b.get_value()

    def get_parameters(self):
        """
        Returns the shared variables of the layer
        :return: dict of shared variables
        """
        return {self.w.name: self.w, self.b.name: self.b}
