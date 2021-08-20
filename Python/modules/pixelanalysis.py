# -*- coding: utf-8 -*-
"""
Functions used for visualization of the tensorflow DL models.

Created on Fri May 21 12:42:52 2021

@author: iverm
"""
import tensorflow as tf
import numpy as np
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



class GuidedReLU(tf.keras.layers.Layer):
    '''
    Guided ReLU activation layer to be used
    if activation functions are applied as layers.
    '''
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        return guided_relu(x)



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
        elif hasattr(layer, 'activation') and layer.activation == 'relu':
            layer.activation = guided_relu
            x = layer(x)
        else:
            x = layer(x)
        
    return Model(inputs = trained_model.inputs, outputs = x)



def compute_gradients(inputs, model, num_class):
    '''
    Compute gradients of model predictions for class num_class wrt inputs.

    Parameters
    ----------
    inputs : tf.Tensor
        Images of shape (num_images, height, width, channels).
    model : tf.Model
        Trained model with output of shape (num_images, num_classes).
    num_class : int
        Which output to compute gradients wrt.

    Returns
    -------
    tf.Tensor
        Gradient images of shape (num_images, height, width, channels).

    '''
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = model(inputs)[:, num_class]

    return tape.gradient(outputs, inputs)
    
    

def build_extractor(model, layers):
    """
    Returns model which outputs hidden feature maps from given layers.
    
    Example of usage:
        
    extractor = build_extractor(model, layers)
    features = extractor.predict(img_tensors)
    plt.imshow(features)
    """

    layer_outputs = [layer.output for layer in [model.layers[i] for i in layers]]

    return Model(inputs = model.input, outputs = layer_outputs)



def generate_path_inputs(baseline_img, input_img, m):
    '''
    Generates m interpolated images between input and baseline image.

    Parameters
    ----------
    baseline_img : numpy.ndarray
        3D tensor of floats.
    input_img : numpy.ndarray
        3D tensor of floats.
    m : int
        Number of path images excluding one endpoint.
        Should be an even number.

    Returns path_inputs
    -------
    4D tf.tensor of step images.

    '''
    if not len(baseline_img.shape) == len(input_img.shape) == 3:
        raise Exception('Input images must have shape (W, H, C)') 
    alphas = np.linspace(0, 1, m + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    delta = np.expand_dims(input_img, 0) - np.expand_dims(baseline_img, 0)
    path_inputs = np.expand_dims(baseline_img, 0) + alphas * delta
    
    return tf.convert_to_tensor(path_inputs)



def integrate_grads(grads):
    '''
    Compute integrated gradients by Composite Simpsons rule.
    Total number of gradient images should be an odd number.
    Assumes that length of interval is 1.
    '''
    h = 1 / (np.shape(grads)[0] - 1)
    sum1 = 4*np.sum(grads[1::2], axis=0)
    sum2 = 2*np.sum(grads[2:-2:2], axis=0)
    return (h / 3) * (grads[0] + grads[-1] + sum1 + sum2)