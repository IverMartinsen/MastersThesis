# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:42:52 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.models import Model



@tf.custom_gradient
def guided_relu(x):
    '''
    Guided ReLU activation function.
    '''
    # nonzero gradient only if gradient and activation is positive
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    # use normal relu function in forward pass
    return tf.nn.relu(x), grad



def build_model(trained_model):
    '''
    Returns model similar to input.
    ReLU activations are replaced by guided ReLU activations.
    '''
    x = trained_model.inputs[0]

    for layer in trained_model.layers[1:]:
        # replace relu activation layers
        if isinstance(layer, tf.keras.layers.ReLU):
            x = tf.keras.layers.Activation(guided_relu)(x)
        # or replace relu activation functions
        #elif hasattr(layer, 'activation') and layer.activation == 'relu':
        #    layer.activation = guided_relu
        #    x = layer(x)
        else:
            x = layer(x)
        
    return Model(inputs = trained_model.inputs, outputs = x)



def compute_grads(inputs, model):
    '''
    Returns gradients of model output wrt inputs.
    '''
    if len(inputs.shape) == 4:
        num_images = 'multiple'
    else:
        inputs = tf.expand_dims(inputs, 0)
        num_images = 'single'
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs)

    if num_images == 'multiple':
        return tape.gradient(outputs, inputs)
    else:
        return tf.squeeze(tape.gradient(outputs, inputs), 0)