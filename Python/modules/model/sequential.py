# -*- coding: utf-8 -*-
"""
Example of using tf.keras.Sequential to build model. 

Created on Fri May 21 11:29:41 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, InputLayer

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.regularizers import L2

Model = tf.keras.Sequential([
    
    InputLayer(input_shape=(128, 128, 1)),
    
    # preprocessing
    Rescaling(1./255.),
    
    # data augmentation
    RandomFlip("horizontal"),
    
    # conv block 1
    Conv2D(8, 3, padding='same', kernel_regularizer=L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    
    # conv block 2
    Conv2D(16, 3, padding='same', kernel_regularizer=L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
        
    # conv block 3
    Conv2D(32, 3, padding='same', kernel_regularizer=L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),

    # conv block 4
    Conv2D(64, 3, padding='same', kernel_regularizer=L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    
    Flatten(),
    
    # dense layer 1
    Dense(32, kernel_regularizer=L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),

    # dense layer 2
    Dense(16, kernel_regularizer=L2(l2=1e-3)),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),
    
    # output layer
    Dense(1, activation='sigmoid', kernel_regularizer=L2(l2=1e-3))
    ])