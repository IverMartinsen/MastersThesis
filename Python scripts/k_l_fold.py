# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:30:59 2021

@author: iverm
"""
import os      
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from imageloader import imageloader

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Input, SeparableConv2D, GlobalAveragePooling2D

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def build_model():
    L2 = tf.keras.regularizers.L2()

    inputs = Input(shape=(128, 128, 1))
    
    #x = RandomZoom(0.2)(inputs)
    #x = RandomRotation(0.2)(x)
    #x = RandomFlip()(x)
    
    x = Rescaling(1./255)(inputs)
    
    x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = MaxPool2D()(x)
    
    x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = MaxPool2D()(x)
    
    x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = MaxPool2D()(x)
    
    x = Flatten()(x)
    x = Dense(64, kernel_regularizer=L2)(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, kernel_regularizer=L2)(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    return model


plt.ioff()

'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

k = 5
sets = imageloader(path, (128, 128), 5, seed=123)

'''
Train model
'''
#sets = ['a', 'b', 'c', 'd', 'e']

destination = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Forsøk 14.06.2021'

folder_name = 'Learning curves'
os.makedirs(destination + '\\' + folder_name, exist_ok=True)


individual_results = pd.DataFrame()
summary_results = pd.DataFrame()
trial_num =0
model = None
for test_ds in sets:
    for valid_ds in (ds for ds in sets if ds != test_ds):
        trial_num += 1
        
        generators = [ds for ds in sets if ds not in (test_ds, valid_ds)]
        
        x_tr = np.concatenate([generator['images'] for generator in generators])
        y_tr = np.concatenate([generator['labels'] for generator in generators])
        
        x_va = valid_ds['images']
        y_va = valid_ds['labels']
        
        tf.keras.backend.clear_session()
        
        del model

        model = build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True)]

        history = model.fit(
            x_tr,
            y_tr,
            epochs=1000,
            validation_data=(x_va, y_va),
            callbacks=callbacks)
            

        '''Plot loss and save figure'''
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.plot(np.argmin(history.history['val_loss']), 
                 np.min(history.history['val_loss']),
                 marker='o',
                 label='Minimum validation loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig(fname=destination + '\\' + folder_name + '\\trial' + str(trial_num))

        '''
        Evaluate on test set
        '''
        
        #model.evaluate(test_ds)
        
        predictions = model.predict(test_ds['images'])
        labels = predictions.round()
        
        
        dataframe = pd.DataFrame(
                test_ds['labels'],
                index=test_ds['filenames'],
                columns=[trial_num])
        
        dataframe[trial_num] = dataframe[trial_num] == labels.flatten()
        
        dataframe = dataframe*1
        
        individual_results = pd.merge(
            individual_results, dataframe, how='outer', left_index=True, right_index=True)
        
        idx = np.where(test_ds['labels'] == 0)

        acc_0 = np.sum(test_ds['labels'][idx] == labels.flatten()[idx]) / len(test_ds['labels'][idx])

        idx = np.where(test_ds['labels'] == 1)

        acc_1 = np.sum(test_ds['labels'][idx] == labels.flatten()[idx]) / len(test_ds['labels'][idx])

        
        dataframe = pd.DataFrame(
                [model.evaluate(test_ds['images'], test_ds['labels'])[1], acc_0, acc_1],
                index=['Accuracy', 'cc', 'neac'],
                columns=[trial_num])
        
        summary_results = pd.merge(
            summary_results, dataframe, how='outer', left_index=True, right_index=True)

individual_results.to_excel(destination + '\\individual_results.xlsx')
summary_results.to_excel(destination + '\\summary_results.xlsx')

        
idx = np.where(test_ds['labels'] == 0)

np.sum(test_ds['labels'][idx] == labels.flatten()[idx]) / len(test_ds['labels'][idx])