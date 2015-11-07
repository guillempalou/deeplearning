from src.deep_learning import create_theano_tensor
from src.deep_learning import SoftMaxLayer
from src.deep_learning import BaseNet
from src.deep_learning import HiddenLayer


class MLP(BaseNet):

    def __init__(self, name, n_in, n_hidden, n_out):

        self.name = name

        self.hidden_layer = HiddenLayer(name + "_hidden", n_in, n_hidden, activation='tanh', init_weights=null,
                                        init_bias=null, dropout=0)
        self.output_layer = SoftMaxLayer(name + "_output", n_hidden, n_out)

        self.layers = [self.hidden_layer, self.output_layer]

        self.X = create_theano_tensor(name + "_X", 2, float)
        self.Y = create_theano_tensor(name + "_Y", 1, int)

        self.params = self.hidden_layer.params + self.output_layer.params

        super(MLP, self).__init__(name)


    def transform(self, x, mode):
        output_hidden = self.hidden_layer.transform(x, mode)
        output = self.output_layer.transform(output_hidden, mode)
        return output


