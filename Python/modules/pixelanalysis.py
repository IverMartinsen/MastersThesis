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



def generate_path_inputs(baseline_img, input_img, alphas):
    '''
    Generates m interpolated images between input and baseline image.

    Parameters
    ----------
    baseline_img : numpy.ndarray
        3D tensor of floats.
    input_img : numpy.ndarray
        3D tensor of floats.
    alphas : numpy.ndarray
        Sequence of alpha values.    
    
    Returns path_inputs
    -------
    4D tf.tensor of step images.

    '''
    if not len(baseline_img.shape) == len(input_img.shape) == 3:
        raise Exception('Input images must have shape (W, H, C)') 
    delta = np.expand_dims(input_img, 0) - np.expand_dims(baseline_img, 0)
    path_inputs = np.expand_dims(baseline_img, 0) + alphas * delta
    
    return tf.convert_to_tensor(path_inputs)



def integral_approximation(gradients):
    '''
    Approximate integration of input using Riemann sums
    and the trapezoidal rule.

    Parameters
    ----------
    gradients : tf.Tensor
        Can have any shape.

    Returns
    -------
    integrated_gradients : tf.Tensor
        Shape as input.

    '''
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0, dtype=tf.float64)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients



@tf.function
def integrated_gradients(
        baseline, image, target_class_idx, m_steps=50, batch_size=32):
    
    
    # Generate sequence of alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Batch computations for speed, memory efficiency, and scaling.
    # For each range in alphas, compute gradients.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # Generate interpolated images between baseline and input image.
        interpolated_path_input_batch = generate_path_inputs(
            baseline=baseline,
            image=image,
            alphas=alpha_batch)

        # Compute gradients for model output wrt batch of interpolated images. 
        gradient_batch = compute_gradients(
            images=interpolated_path_input_batch,
            target_class_idx=target_class_idx)

        # Write batch indices and gradients into TensorArray.
        gradient_batches = gradient_batches.scatter(
            tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients



def ig_error(baseline, image, integrated_gradients):
    '''
    Returns percentage relative error for the integrated_gradients.
    '''
    aim = tf.math.abs(model(image) - model(baseline))
    result = tf.math.reduce_sum(integrated_gradients)
    return 100 * tf.math.abs(result - aim) / aim