import sys

sys.path.extend(['/Users/guillem/developer/kaggle/deep_learning'])

import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from skimage.io import imread
from deep_learning.common.tensors import create_theano_tensor
from deep_learning.layers.convolutional import ConvolutionalLayer
import numpy as np

img = imread('/Users/guillem/developer/kaggle/digit/resources/test.jpg').astype(np.float32) / 255

# define the inputs
input_shape = np.asarray((3, img.shape[0], img.shape[1]))
n_filters = 20
filters = np.asarray((n_filters, 5, 5))

minibatch = 1
batch = np.zeros([minibatch] + list(input_shape.astype(np.int)), dtype=np.float32)

for ch in range(3):
    batch[0, ch, :, :] = img[:, :, 0]

pool = np.asarray((1, 1))
X = create_theano_tensor("conv_test", 4, float)

layer = ConvolutionalLayer("test",
                           input=input_shape,
                           filters=filters,
                           pool=pool,
                           stride=(7, 11),
                           activation='relu')

f = theano.function([X], layer.transform(X, minibatch=1))

output = f(batch)

for f in range(n_filters):
    plt.subplot(4, 5, f + 1)
    plt.imshow(output[0, f, :, :])

plt.show()
