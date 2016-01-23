import logging
import theano.tensor as T

from deep_learning.layers.linear_unit_layer import LinearUnitLayer


class HiddenLayer(LinearUnitLayer):

    logger = logging.getLogger("HiddenLayer")

    def __init__(self, name, **kwargs):
        super(HiddenLayer, self).__init__(name=name, **kwargs)
        self.logger.debug("Creating hidden layer {0}".format(name))
        self.logger.debug("Layer with {0} inputs and {1} outputs".format(self.in_shape, self.out_shape))

        # set the activation function for this layer
        # the activation is set to the identity on the parent class
        self.activation = kwargs.get("activation", self.activation)

        self.logger.debug("Activation {0} - {1} inputs and {2} outputs".format(self.activation,
                                                                               self.in_shape,
                                                                               self.out_shape))