from deep_learning.common.tensors import create_theano_tensor
from deep_learning.layers.softmax import SoftMaxLayer
from deep_learning.nets.base_net import BaseNet
from deep_learning.layers.hidden_layer import HiddenLayer


class MLP(BaseNet):

    def __init__(self, name, n_in, n_hidden, n_out):

        self.name = name

        self.hidden_layer = HiddenLayer(name + "_hidden", n_in, n_hidden, activation='tanh')
        self.output_layer = SoftMaxLayer(name + "_output", n_hidden, n_out)

        self.layers = [self.hidden_layer, self.output_layer]

        self.X = create_theano_tensor(name + "_X", 2, float)
        self.Y = create_theano_tensor(name + "_Y", 1, int)

        self.params = self.hidden_layer.params + self.output_layer.params

        super(MLP, self).__init__(name)


    def transform(self, x):
        output_hidden = self.hidden_layer.transform(x)
        output = self.output_layer.transform(output_hidden)
        return output


