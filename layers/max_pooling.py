from deep_learning.layers.base_layer import BaseLayer

class MaxPoolingLayer(BaseLayer):

    def __init__(self, name, n_in, n_out, in_type=float, truth_type=float):
        super(MaxPoolingLayer, self).__init__(None, None, None)


    def transform(self, x):
        pass