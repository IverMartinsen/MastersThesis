# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:29:41 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import InputLayer

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

Model = tf.keras.Sequential([
    InputLayer(input_shape=(128, 128, 1)),
    Rescaling(1./127.5, offset=-1),
    RandomFlip("horizontal"),
    Conv2D(8, 3, strides = (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    
    Conv2D(16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
        
    Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),

    Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    
    Flatten(),
    
    Dense(32, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),

    Dense(16, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),

    Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    ])

@tf.custom_gradient
def guidedReLU(x):
    # nonzero gradient only if gradient and activation is positive
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    # use normal relu function in forward pass
    return tf.nn.relu(x), grad

class GuidedReLU(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        return guidedReLU(x)

GBModel = tf.keras.Sequential([
    InputLayer(input_shape=(128, 128, 1)),
    Rescaling(1./127.5, offset=-1),
    RandomFlip("horizontal"),
    Conv2D(8, 3, strides = (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    GuidedReLU(),
    MaxPool2D(),
    
    Conv2D(16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    GuidedReLU(),
    MaxPool2D(),
        
    Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    GuidedReLU(),
    MaxPool2D(),

    Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    GuidedReLU(),
    MaxPool2D(),
    
    Flatten(),
    
    Dense(32, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    GuidedReLU(),
    Dropout(0.5),

    Dense(16, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),
    BatchNormalization(),
    GuidedReLU(),
    Dropout(0.5),

    Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    ])