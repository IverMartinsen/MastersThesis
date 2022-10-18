"""
Script for a single training run on the cod otolith data.
Includes visualization by guided backpropagation.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from modules.model.functional import model
from modules.analysis.guided_backpropagation import build_gb_model_seq as build_gbmodel
from modules.analysis.utils import compute_gradients
from matplotlib import cm


'''
Import images
'''
path = (r'C:\Users\iverm\Google Drive\Masteroppgave' + 
        r'\Data\Torskeotolitter\standard')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)


'''
Train model
'''

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
]

history = model.fit(train_ds,
                    epochs=100,
                    validation_data=valid_ds,
                    callbacks=callbacks)


'''
Plot loss
'''
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(
    np.argmin(history.history['val_loss']), 
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
plt.plot(
    np.argmin(history.history['val_loss']),
    history.history[
        'val_binary_accuracy'][np.argmin(history.history['val_loss'])],
    marker='o',
    label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')


'''
Save model and weights to a folder given by cp_location
'''
cp_location = 'path_to_checkpoints'

model.save(
    cp_location + '\\' + os.path.split(path)[-1], overwrite=False)
model.save_weights(
    cp_location + '\\' + os.path.split(path)[-1] + '_weights\\trained')


'''
Visualization of pixel relevance by guided backpropagation
'''
guided_backprop_model = build_gbmodel(model)

image = list(valid_ds)[0][0][3]
grads = compute_gradients(image, guided_backprop_model, num_class=0)

vmin = 0
vmax = 1

hm = (grads - np.min(grads)) / np.max(grads - np.min(grads))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(image, 'gray')
ax1.axis('off')
ax1.set_title('Input image')

ax2.imshow(hm, cm.jet, vmin=vmin, vmax=vmax)
ax2.axis('off')
ax2.set_title('Pixel relevance by Guided Backpropagation')

ax3.imshow(image, 'gray', alpha=0.5)
ax3.imshow(hm, cm.jet, vmin=vmin, vmax=vmax, alpha=0.7)
ax3.axis('off')
ax3.set_title('Combined image')
