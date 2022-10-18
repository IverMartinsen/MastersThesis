# -*- coding: utf-8 -*-
"""
Example of applying random search procedures to optimize hyperparameters.

Created on Tue Jun 15 10:24:26 2021

@author: iverm
"""

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
import kerastuner as kt


def build_model(hp):
    """
    Model building function to be used with hyperparameters tuning.
    Samples following hyperparameters:
        L2 norm penalty
        #convolution blocks
        #filters for each block
        #layers for each block
        #dense layers
        #neurons for each dense layer
        dropout rate for the first dense layer
        learning rate

    Parameters
    ----------
    hp : keras_tuner.HyperParameters
        Hyperparameters container.

    Returns
    -------
    model : tf.keras.Model
        Compiled tensorflow model.

    """
    L2 = tf.keras.regularizers.L2(
        l2=hp.Float("l2_norm", 1e-5, 1e-1, sampling="log"))
    
    inputs = Input(shape=(128, 128, 1))
    
    x = Rescaling(1./255)(inputs)

    for i in range(hp.Int("conv_blocks", 3, 5)):

        filters = hp.Choice(
                "conv_block_" + str(i+1), (8, 16, 24, 32, 48, 64, 96, 128))

        for j in range(hp.Int("block_size_" + str(i+1), 1, 2)):
            
            x = Conv2D(filters, 3, padding='same', kernel_regularizer=L2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        
        x = MaxPool2D()(x)
        
    x = Flatten()(x)
    
    for i in range(hp.Int("dense_blocks", 1, 2)):
        
        x = Dense(hp.Choice(
                "dense_block_" + str(i+1), (8, 16, 24, 32, 48, 64, 96, 128)), 
                activation='relu', kernel_regularizer=L2)(x)
        
        if i == 0:
            x = Dropout(hp.Float("dropout", 0, 0.5, 0.1))(x)
  
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=L2)(x)
    
    _model = tf.keras.Model(inputs, outputs)
    
    _model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-1, sampling="log")),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    
    return _model


# import training and validation data
path = r'C:\Users\iverm\Google Drive\Masteroppgave\Data\Torskeotolitter\standard_convex'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.4,
    subset="training",
    seed=123,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.4,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)

# define parameter tuner instance
tuner = kt.RandomSearch(
    build_model,
    objective="val_binary_accuracy",
    max_trials=200, 
    project_name='random_search')

# search for parameters
tuner.search(
    train_ds, 
    epochs=100, 
    validation_data=valid_ds,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])

# save best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
model.save(r'C:\Users\iverm\Google Drive\Masteroppgave\convex_model\best_convex_model')
