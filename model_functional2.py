# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:19:28 2021

@author: iverm
"""
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import SeparableConv2D, MaxPool2D, Add, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf

def build_model():
    
    inputs = Input(shape=(128, 128, 1))
    x = Rescaling(1./255)(inputs)
    x = RandomFlip("horizontal")(x)

    # Entry block
    x = Conv2D(8, 3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)


    x = Conv2D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)

    previous_block_activation = x  # Set aside residual

    for size in [32, 64]:
        x = ReLU()(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = ReLU()(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPool2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(1024, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)
    
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)