# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:29:41 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Add
from tensorflow.keras.layers import Input, SeparableConv2D, GlobalAveragePooling2D

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
'''
L2 = tf.keras.regularizers.L2(l2=1e-5)

tf.keras.backend.clear_session()

inputs = Input(shape=(128, 128, 1))
x = Rescaling(1./255)(inputs)
x = RandomFlip("horizontal")(x)

x = Conv2D(40, 3, strides = (1, 1), padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Conv2D(64, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)
    
x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Conv2D(24, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Flatten()(x)

x = Dense(30, kernel_regularizer=L2)(x)
#x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.1)(x)

#x = Dense(16, kernel_regularizer=L2)(x)
#x = BatchNormalization()(x)
#x = ReLU()(x)
#x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid', kernel_regularizer=L2)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
'''
def build_model():
    L2 = tf.keras.regularizers.L2()

    inputs = Input(shape=(128, 128, 1))

    x = Rescaling(1./255)(inputs)
    
    x = Conv2D(8, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(8, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPool2D()(x)
    
    x = Conv2D(16, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(16, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPool2D()(x)

    x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPool2D()(x)
    
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=L2)(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    return model