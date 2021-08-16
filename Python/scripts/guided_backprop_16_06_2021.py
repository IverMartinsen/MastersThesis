# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:36:32 2021

@author: iverm
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
from scipy.signal import convolve2d
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from imageloader import imageloader
from guided_backprop import build_model as build_gb_model
from guided_backprop import compute_grads

def build_model():
    L2 = tf.keras.regularizers.L2()

    inputs = Input(shape=(128, 128, 1))
    
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
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, kernel_regularizer=L2)(x)
    x = ReLU()(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    return model

path = r'C:\Users\iverm\Google Drive\Masteroppgave\Data\Torskeotolitter\standard'

k = 5
sets = imageloader(path, (128, 128), 5, seed=123)

[sets[i] for i in (0, 1, 4)]

x_tr = np.concatenate([generator['images'] for generator in [sets[i] for i in [0, 1, 4]]])
y_tr = np.concatenate([generator['labels'] for generator in [sets[i] for i in [0, 1, 4]]])

x_va = sets[2]['images']
y_va = sets[2]['labels']

x_te = sets[3]['images']
y_te = sets[3]['labels']

model = build_model()

callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]

history = model.fit(
    x_tr,
    y_tr,
    epochs=1000,
    validation_data=(x_va, y_va),
    callbacks=callbacks)

model.evaluate(x_te, y_te)

gb_model = build_gb_model(model)


gb_model_2 = tf.keras.Model(inputs=gb_model.inputs, outputs=[1-output for output in gb_model.outputs])

grads = tf.convert_to_tensor(np.zeros_like(x_te[0]))

pred = model.predict(x_te).round().flatten()

tensor = tf.convert_to_tensor(x_te[np.where(pred == 0)])

with tf.GradientTape() as tape:
    tape.watch(tensor)
    outputs = gb_model_2(tensor)
    
grads = tape.gradient(outputs, tensor)

mean_grads = np.mean(grads, 0)

from image_processing.gaussian_kernel import gaussian_kernel
kernel = gaussian_kernel(1)

test = convolve2d(mean_grads, kernel, mode='same')

mean_grads_standard = (test - np.min(test)) / np.max((test - np.min(test)))

pred = model.predict(x_te).round()[np.where(y_te == 1)].flatten()

labs = y_te[np.where(y_te == 1)].flatten()

np.sum(pred == labs)

plt.imshow(mean_grads_standard, cm.jet, vmin=0.4, vmax=0.8)
plt.axis('off')
plt.show()

layer_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[5].output)
layer_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[9].output)
layer_3 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[13].output)


tensor = tf.convert_to_tensor(x_te)

layer1_output = np.mean(layer_1(tensor), axis = 0)

layer2_output = np.mean(layer_2(tensor), axis = 0)
layer3_output = np.mean(layer_3(tensor), axis = 0)

fig = plt.figure()
gridspec.GridSpec(3, 32)

for i in range(16):
    plt.subplot2grid((3,32), (0,i*2), colspan=2, rowspan=1)
    plt.imshow(layer2_output[:, :, i], cm.afmhot)
    plt.axis('off')
    
for i in range(16):
    plt.subplot2grid((3,32), (1,i*2), colspan=2, rowspan=1)
    plt.imshow(layer2_output[:, :, 16+i], cm.afmhot)
    plt.axis('off')
    
for i in range(32):
    plt.subplot2grid((3,32), (2,i), colspan=1, rowspan=1)
    plt.imshow(layer3_output[:, :, i], cm.afmhot)
    plt.axis('off')
    