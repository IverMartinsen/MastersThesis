# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:05:28 2021

@author: iverm
"""
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import SeparableConv2D, MaxPool2D, Add, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf

def build_model(hp):
    
    #L2 = tf.keras.regularizers.L2(
    #    l2=hp.Float("l2_norm", 1e-5, 1e-2, sampling="log"))
    
    inputs = Input(shape=(128, 128, 1))
    
    #x = RandomFlip("horizontal")(inputs)
    x = Rescaling(1./255)(inputs)
    
    x = Conv2D(hp.Int("filters_1", 8, 64, step=8), 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Conv2D(hp.Int("filters_2", 8, 64, step=8), 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    previous_block_activation = x
    
    filters = hp.Int("filters_3", 8, 64, step=8)
    
    x = SeparableConv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
        
    residual = Conv2D(filters, 1, padding="same")(previous_block_activation)
    residual = MaxPool2D()(residual)
    
    x = Add()([x, residual])
    
    previous_block_activation = x
    
    filters = hp.Int("filters_4", 8, 64, step=8)
    
    x = SeparableConv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
        
    residual = Conv2D(filters, 1, padding="same")(previous_block_activation)
    residual = MaxPool2D()(residual)
    
    x = Add()([x, residual])
    
    x = SeparableConv2D(hp.Int("filters_5", 8, 64, step=8), 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)
    x = Dense(hp.Int("hidden_size", 10, 100, step=10))(x)
    x = Dropout(0.5)(x)
  
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    
    return model