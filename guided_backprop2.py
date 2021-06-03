# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:12:45 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import InputLayer

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

model.save_weights('test')

from standard_model import GBModel


test_model = GBModel

test_model.load_weights('test')

inputs = list(test_ds)[0][0][0:10]

with tf.GradientTape() as tape:
    tape.watch(inputs)
    outputs = test_model(inputs)

grads = tape.gradient(outputs, inputs)

from matplotlib import cm

hm = (grads[0] - np.min(grads[0])) / np.max(grads[0] - np.min(grads[0]))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(inputs[0], 'gray')
ax1.axis('off')
ax1.set_title('Coastal Cod no. 226')

ax2.imshow(hm, 'gray')
ax2.axis('off')
ax2.set_title('Pixel relevance by Guided Backpropagation')

ax3.imshow(inputs[0], 'gray', alpha = 0.5)
ax3.imshow(hm, cm.jet, vmin = 0.4, vmax = 0.5, alpha = 0.7)
ax3.axis('off')
ax3.set_title('Combined image')
