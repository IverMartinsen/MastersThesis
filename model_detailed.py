import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model

class CNN(Model):
  
  def __init__(self):
    
    super().__init__()

    self.scale = tf.keras.layers.experimental.preprocessing.Rescaling(scale = 1. / 255)
    
    self.conv1 = Conv2D(8, 3,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2))
    self.pool1 = MaxPool2D((2, 2), strides=(2, 2))
    self.batchnorm1 = BatchNormalization()
    
    self.conv2 = Conv2D(16, 3,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2))
    self.pool2 = MaxPool2D((2, 2), strides=(2, 2))
    self.batchnorm2 = BatchNormalization()
    
    self.conv3 = Conv2D(32, 3,
                        activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2))
    self.pool3 = MaxPool2D((2, 2), strides=(2, 2))
    self.batchnorm3 = BatchNormalization()
    
    self.flatten = Flatten()
    self.dense1 = Dense(32,
                        activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2))
    self.batchnorm4 = BatchNormalization()
    self.dense2 = Dense(2,
                        activation='sigmoid',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2))

    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    self.optimizer = tf.keras.optimizers.Adam()
    
    self.train_loss = tf.keras.metrics.Mean()
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    self.valid_loss = tf.keras.metrics.Mean()
    self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  @tf.function
  def call(self, x):

    x = self.scale(x)
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.batchnorm1(x)
    
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.batchnorm2(x)
    
    x = self.conv3(x)
    x = self.pool3(x)
    x = self.batchnorm3(x)
    
    x = self.flatten(x)
    
    x = self.dense1(x)
    x = self.batchnorm4(x)
    
    return self.dense2(x)

  @tf.function
  def training_step(self, images, labels):
    
    with tf.GradientTape() as tape:
      y_pred = self.call(images)
      loss = self.loss_object(labels, y_pred)
    
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
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
            f'   Training loss:       {self.training_loss_history[i].round(4)}\n'
            f'   Training accuracy:   {self.train_accuracy.result().numpy()}\n'
            f'   Validation loss:     {self.validation_loss_history[i].round(4)}\n'
            f'   Validation accuracy: {self.valid_accuracy.result().numpy()}')
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
    
    y_pred = np.concatenate([tf.math.argmax(self.call(images), axis = 1) for images, labels in test_ds], axis=0)


    accuracy = accuracy.result().numpy()

    return {'labels': y_pred, 'accuracy': accuracy}

  def label(self, dataset):

    return tf.argmax(self.predict(dataset), axis = 1)