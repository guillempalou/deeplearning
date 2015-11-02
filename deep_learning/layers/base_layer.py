class BaseLayer(object):

    def __init__(self, **kwargs):
        self.parameters = kwargs

        # set some basic variables that all inputs need to have
        self.name = self.parameters["name"]
        self.in_shape = self.parameters["in_shape"]
        self.out_shape = self.parameters["out_shape"]

    def transform(self, x, mode='train'):
        raise NotImplementedError("No method defined")

    def __str__(self):
        return self.parameters
