from src.deep_learning.augmentation.base_augmentation import BaseAugmentation

class IdentityAugmentation(BaseAugmentation):

    def __init__(self):
        super(IdentityAugmentation, self).__init__()
        self.n_dups = 1

    def augment(self, x, y):
        return x, y