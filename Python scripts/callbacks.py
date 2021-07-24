# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:02:43 2021

@author: iverm
"""
import tensorflow as tf

class TestAccuracy(tf.keras.callbacks.Callback):
    """Custom callback for monitoring test accuracy during training."""
    
    def __init__(self, test_ds):
        super().__init__()
        
        self.test_ds = test_ds
        self.accuracy = []
            
    def on_epoch_end(self, epoch, logs=None):
        accuracy = self.model.evaluate(self.test_ds, verbose=0)[1]
        self.accuracy.append(accuracy)

    def on_train_end(self, logs=None):
        self.model.history.history['test_binary_accuracy'] = self.accuracy