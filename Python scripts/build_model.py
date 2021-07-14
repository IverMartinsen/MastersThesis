# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:48:11 2021

@author: iverm
"""

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf

def build_model(hp):
    
    L2 = tf.keras.regularizers.L2(
        l2=hp.Float("l2_norm", 1e-5, 1e-1, sampling="log"))
    
    inputs = Input(shape=(128, 128, 1))

    x = Rescaling(1./255)(inputs)

    for i in range(hp.Int("conv_blocks", 3, 5)):        
        for j in range(hp.Int("block_size", 1, 2)):
        
            x = Conv2D(
                hp.Int("block_" + str(i) + "conv_" + str(j), 8, 128, step=8), 
                3,
                padding='same',
                kernel_regularizer=L2
                )(x)
            
            x = BatchNormalization()(x)
            x = ReLU()(x)
        
        x = MaxPool2D()(x)
        
    x = Flatten()(x)
    
    for i in range(hp.Int("dense_blocks", 1, 2)):
        
        x = Dense(hp.Int("dense_" + str(i), 8, 128, step=8),
                  activation='relu',
                  kernel_regularizer=L2
                  )(x)
        x = Dropout(0.5)(x)
  
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=L2)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-1, sampling="log")),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    
    return model