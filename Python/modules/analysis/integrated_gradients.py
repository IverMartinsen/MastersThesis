# -*- coding: utf-8 -*-
"""
Module for computing integrated gradients.

Example of usage:
    
    import integrated_gradients
    
    int_grads = integrated_gradients(
                         model,
                         baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):


Created on Wed Aug 25 13:54:27 2021

@author: iverm
"""

import tensorflow as tf
from modules.analysis.utils import compute_gradients


def generate_path_inputs(baseline, image, alphas):
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
    if not len(baseline.shape) == len(image.shape) == 3:
        raise Exception('Input images must have shape (W, H, C)') 
    
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.cast(tf.expand_dims(baseline, 0), tf.float32)
    input_x = tf.cast(tf.expand_dims(image, 0), tf.float32)

    delta = input_x - baseline_x
    path_inputs = baseline_x + alphas_x * delta
    
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
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(
        2.0, dtype=gradients.dtype)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients



@tf.function
def integrated_gradients(model,
                         baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
    '''
    Compute integrated gradients for input image wrt baseline image.

    Parameters
    ----------
    model : tf.keras.models.Model
        Trained keras model.
    baseline : numpy.ndarray
        Baseline image.
    image : numpy.ndarray
        Input image.
    target_class_idx : int
        Index for target class.
    m_steps : int, optional
        Number of integration steps. The default is 50.
    batch_size : int, optional
        Batch size. The default is 32.

    Returns
    -------
    integrated_gradients : tf.Tensor
        Tensor with shape as input image.

    '''
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
            inputs=interpolated_path_input_batch,
            model=model, 
            num_class=target_class_idx)

        # Write batch indices and gradients into TensorArray.
        gradient_batches = gradient_batches.scatter(
            tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = tf.cast(
        image - baseline, avg_gradients.dtype) * avg_gradients

    return integrated_gradients



def ig_error(model, baseline, image, integrated_gradients, target_class_idx):
    '''
    Returns percentage relative error for the integrated_gradients.
    '''
    aim = tf.math.abs(
        model(
            image[tf.newaxis, :, :, :])[:, target_class_idx] - 
            model(baseline[tf.newaxis, :, :, :])[:, target_class_idx])
    result = tf.math.reduce_sum(integrated_gradients)
    return 100 * tf.math.abs(result - aim) / aim