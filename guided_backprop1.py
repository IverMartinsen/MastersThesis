# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:42:52 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.models import Model


inputs = list(test_ds)[0][0][0:10]

@tf.custom_gradient
def guidedReLU(x):
    # nonzero gradient only if gradient and activation is positive
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    # use normal relu function in forward pass
    return tf.nn.relu(x), grad

GuidedReLU = tf.keras.layers.Activation(guidedReLU)

x = model.layers[0].output

for layer in model.layers[:-1]:
    if isinstance(layer, tf.keras.layers.ReLU):
        print('true')
        x = GuidedReLU(x)
    else:
        x = layer(x)

x = model.layers[-1](x)

# define model to be modified to using guidedReLU
guided_backprop_model = Model(
    inputs = [model.inputs],
    outputs = x
)

guided_backprop_model.layers[-1].activations = tf.keras.activations.linear

with tf.GradientTape() as tape:
    tape.watch(inputs)
    outputs = guided_backprop_model(inputs)

grads = tape.gradient(outputs, inputs)

import matplotlib.pyplot as plt
plt.imshow(grads[0], 'gray')
