# -*- coding: utf-8 -*-
"""
Example of building models with tf.keras.Functional.

Created on Fri May 21 11:29:41 2021

@author: iverm
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Input

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

L2 = tf.keras.regularizers.L2(l2=1e-5)

inputs = Input(shape=(128, 128, 1))

# preprocessing
x = Rescaling(1./255)(inputs)

# data augmentation
x = RandomFlip("horizontal")(x)

# conv block 1
x = Conv2D(40, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

# conv block 2
x = Conv2D(64, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

# conv block 3
x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

# conv block 4
x = Conv2D(24, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D()(x)

x = Flatten()(x)

# dense layer 1
x = Dense(30, kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.1)(x)

# dense layer 2
x = Dense(16, kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.5)(x)

outputs = Dense(1, activation='sigmoid', kernel_regularizer=L2)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)