import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Input, SeparableConv2D, GlobalAveragePooling2D

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard_convex'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.4,
    subset="training",
    seed=543,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.4,
    subset="validation",
    seed=543,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)

'''
Train model
'''
tf.keras.backend.clear_session()
del model

L2 = tf.keras.regularizers.L2(0)

inputs = Input(shape=(128, 128, 1))

#x = RandomZoom(0.2)(inputs)
#x = RandomRotation(0.2)(x)
#x = RandomFlip()(x)

x = Rescaling(1./255)(inputs)

x = Conv2D(8, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPool2D()(x)

x = Conv2D(16, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPool2D()(x)

x = Conv2D(32, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPool2D()(x)

x = Conv2D(64, 3, padding='same', kernel_regularizer=L2)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = MaxPool2D()(x)

x = Flatten()(x)
x = Dense(64, kernel_regularizer=L2)(x)
#x = BatchNormalization()(x)
x = ReLU()(x)
#x = Dropout(0.5)(x)

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

callbacks = [tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True)]

history = model.fit(train_ds,
                    epochs=1000,
                    validation_data=valid_ds,
                    callbacks=callbacks)

model.evaluate(valid_ds)

'''
Plot loss
'''
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(np.argmin(history.history['val_loss']), 
         np.min(history.history['val_loss']),
         marker='o',
         label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')

'''
Plot accuracy
'''
plt.figure(figsize=(12, 8))
plt.plot(history.history['binary_accuracy'], label='Training accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Validation accuracy')
#plt.plot(history.history['test_binary_accuracy'], label='Test accuracy')
plt.plot(np.argmin(history.history['val_loss']), 
         history.history['val_binary_accuracy'][
             np.argmin(history.history['val_loss'])],
         marker='o',
         label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')




