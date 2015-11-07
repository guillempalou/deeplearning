class BaseLayer(object):

    def __init__(self, **kwargs):
        # set some basic variables that all inputs need to have
        self.name = kwargs["name"]
        self.in_shape = kwargs["in_shape"]
        self.out_shape = kwargs["out_shape"]

    def transform(self, x, **kwargs):
        raise NotImplementedError("No method defined")

    def __str__(self):
        return "Layer Shape {0} x {1}".format(self.in_shape, self.out_shape)
