# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:25:20 2021

@author: iverm
"""
import tensorflow as tf
import kerastuner as kt

from standard_model_functional import build_model
from dataloader import dataloader

# import training and validation data
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

train_ds, valid_ds = dataloader(path, (128, 128), 1, [0.7, 0.3])
train_ds = train_ds.shuffle(1000)

# define parameter tuner instance
tuner = kt.tuners.RandomSearch(
    build_model,
    objective='val_binary_accuracy', 
    max_trials=20, 
    executions_per_trial=1,
    project_name='grid_search')

# search for parameters
tuner.search(
    train_ds, 
    epochs=1000, 
    validation_data=valid_ds,
    callbacks = [tf.keras.callbacks.EarlyStopping(
        patience=50, restore_best_weights=True)])