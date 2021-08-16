# -*- coding: utf-8 -*-
"""
Example of subclassing tf.keras.Model and defining model methods manually.

Created on Fri May 21 12:42:52 2021

@author: iverm
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.regularizers import L2

class CNN(Model):
  
    def __init__(self):
      
        super().__init__()
          
        self.scale = Rescaling(scale=1./255)
        
        self.block1_conv = Conv2D(8, 3, kernel_regularizer=L2(l2=1e-2))
        self.block1_norm = BatchNormalization()
        self.block1_relu = ReLU()
        self.block1_pool = MaxPool2D((2, 2), strides=(2, 2))
        
        self.block2_conv = Conv2D(16, 3, kernel_regularizer=L2(l2=1e-2))
        self.block2_norm = BatchNormalization()
        self.block2_relu = ReLU()
        self.block2_pool = MaxPool2D((2, 2), strides=(2, 2))
        
        self.block3_conv = Conv2D(32, 3, kernel_regularizer=L2(l2=1e-2))
        self.block3_norm = BatchNormalization()
        self.block3_relu = ReLU()
        self.block3_pool = MaxPool2D((2, 2), strides=(2, 2))
        
        self.flatten = Flatten()
        
        self.dense1 = Dense(
            32, activation='relu', kernel_regularizer=L2(l2=1e-2))
          
        self.dense2 = Dense(
            1, activation='sigmoid', kernel_regularizer=L2(l2=1e-2))
          
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy()
        
        self.valid_loss = tf.keras.metrics.Mean()
        self.valid_accuracy = tf.keras.metrics.BinaryAccuracy()

    @tf.functional
    def call(self, x):
    
        x = self.scale(x)
        
        x = self.block1_conv(x)
        x = self.block1_norm(x)
        x = self.block1_relu(x)
        x = self.block1_pool(x)
        
        x = self.block2_conv(x)
        x = self.block2_norm(x)
        x = self.block2_relu(x)
        x = self.block2_pool(x)
        
        x = self.block3_conv(x)
        x = self.block3_norm(x)
        x = self.block3_relu(x)
        x = self.block3_pool(x)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        
        return self.dense2(x)

    @tf.function
    def training_step(self, images, labels):
      
        with tf.GradientTape() as tape:
            y_pred = self.call(images)
            loss = self.loss_object(labels, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        
        self.train_accuracy.update_state(labels, y_pred)

    @tf.function
    def test_step(self, images, labels):
      
        y_pred = self.call(images)
        
        valid_loss = self.loss_object(labels, y_pred)
        
        self.valid_loss.update_state(valid_loss)
        
        self.valid_accuracy.update_state(labels, y_pred)

    def train(self, train_ds, valid_ds, epochs):
      
        self.validation_loss_history = np.zeros(epochs)
        self.training_loss_history = np.zeros(epochs)
      
        counter = 0
      
        for i in range(epochs):
      
            for images, labels in train_ds:
                self.training_step(images, labels)
            
            for images, labels in valid_ds:
                self.test_step(images, labels)
          
            self.training_loss_history[i] = self.train_loss.result()
            self.validation_loss_history[i] = self.valid_loss.result()
      
            counter += 1
    
            if counter == 10 or i < 10 or i == (epochs - 1):
                print(
                    f'{i + 1} epochs:\n'
                    f'   Training loss:       ',
                    f'{self.training_loss_history[i].round(4)}\n'
                    f'   Training accuracy:   ',
                    f'{self.train_accuracy.result().numpy()}\n'
                    f'   Validation loss:     ',
                    f'{self.validation_loss_history[i].round(4)}\n'
                    f'   Validation accuracy: ',
                    f'{self.valid_accuracy.result().numpy()}')
                if i != 0 and i != epochs:
                    counter = 0
      
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.valid_loss.reset_states()
        self.valid_accuracy.reset_states()
  
    def test(self, test_ds):
      
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        
        for images, labels in test_ds:
            y_pred = self.call(images)
            accuracy.update_state(labels, y_pred)
        
        y_pred = np.concatenate(
            [tf.math.argmax(
                self.call(
                    images), axis = 1) for images, labels in test_ds], axis=0)
      
      
        accuracy = accuracy.result().numpy()
      
        return {'labels': y_pred, 'accuracy': accuracy}

    def label(self, dataset):
    
        return tf.argmax(self.predict(dataset), axis = 1)