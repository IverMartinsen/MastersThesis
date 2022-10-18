# -*- coding: utf-8 -*-
"""
Utility functions for feature relevance analysis.

Created on Wed Aug 25 13:57:39 2021

@author: iverm
"""
import tensorflow as tf



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

    return tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)


