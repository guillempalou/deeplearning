from deep_learning.common.tensors import create_theano_tensor
from deep_learning.layers.softmax import SoftMaxLayer
from deep_learning.nets.base_net import BaseNet


class Logistic(BaseNet):
    def __init__(self, name, n_in, n_out):
        self.name = name
        self.output_layer = SoftMaxLayer(name + "_output", n_in, n_out)
        self.layers = [self.output_layer]

        # define the parameters
        self.params = [self.output_layer.w, self.output_layer.b]

        # define inputs and outputs based on the layers
        self.X = create_theano_tensor('logistic_gt', 2, float)
        self.Y = create_theano_tensor('logistic_gt', 1, int)

        # call the base class
        super(Logistic, self).__init__(name)

    def transform(self, x):
        return self.output_layer.transform(x)


