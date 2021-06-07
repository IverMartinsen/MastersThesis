# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:29:41 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Input

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

L2 = tf.keras.regularizers.L2(l2=1e-2)

inputs = Input(shape=(128, 128, 1))
x = Rescaling(1./127.5, offset=-1)(inputs)
x = RandomFlip("horizontal")(x)

x = Conv2D(8, 3, strides = (1, 1), padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Conv2D(16, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)
    
x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Conv2D(64, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Flatten()(x)

x = Dense(32, kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.5)(x)

x = Dense(16, kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid', kernel_regularizer=L2)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

def build_model(hp):
    L2 = tf.keras.regularizers.L2(l2=hp.Choice(
        'L2', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))

    inputs = Input(shape=(128, 128, 1))
    x = Rescaling(1./127.5, offset=-1)(inputs)
    x = RandomFlip("horizontal")(x)
    
    x = Conv2D(8, 3, strides = (1, 1), padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Conv2D(16, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
        
    x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Conv2D(64, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Flatten()(x)
    
    x = Dense(32, kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(16, kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=L2)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice(
                'learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    return model