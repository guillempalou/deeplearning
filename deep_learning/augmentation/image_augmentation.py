import abc
import logging
from deep_learning.augmentation.base_augmentation import BaseAugmentation


class ImageAugmentation(BaseAugmentation):

    __metaclass__ = abc.ABCMeta

    logger = logging.getLogger("ImageAugmentation")

    def __init__(self):
        BaseAugmentation.__init__(self)

