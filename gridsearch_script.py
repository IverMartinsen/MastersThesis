# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:25:20 2021

@author: iverm
"""
import tensorflow as tf
import kerastuner as kt

from model_functional3 import build_model
from dataloader import dataloader

# import training and validation data
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

train_ds, valid_ds = dataloader(path, (128, 128), 1, [0.6, 0.4])
train_ds = train_ds.shuffle(366)

# define parameter tuner instance
tuner = kt.tuners.RandomSearch(
    build_model,
    objective='val_loss', 
    max_trials=50, 
    executions_per_trial=1,
    project_name='grid_search')

# search for parameters
tuner.search(
    train_ds, 
    epochs=1000, 
    validation_data=valid_ds,
    callbacks = [tf.keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True)])