from data_generation.augmentation import BaseAugmentation

class IdentityAugmentation(BaseAugmentation):

    def __init__(self):
        super(IdentityAugmentation, self).__init__()
        self.n_dups = 1

    def augment(self, x, y):
        return x, y