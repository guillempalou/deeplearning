import os
import yaml
import numpy as np

from skimage.data import lena
from skimage.transform import resize

from deep_learning.factories.net_factory import create_forward_net_from_dict
from deep_learning.io.data_utils import data_dir

import theano.tensor as T
import matplotlib.pyplot as plt

net_definition = yaml.load(open(os.path.join(data_dir, "configs", "convolutional.yaml")))
net = create_forward_net_from_dict("ConvNet", net_definition["ConvNet"])

img = resize(lena(), (256, 256))
t_img = img[np.newaxis, :, :, :].astype(np.float32)
t_img = np.rollaxis(t_img, 3, 1)

net.setup_test_function()

result = net.test(t_img)
result = np.rollaxis(result, 1, 4)
print(t_img.shape)
print(img.shape)
print(result.shape)
plt.imshow(result[0, ...])
plt.show()
