# -*- coding: utf-8 -*-
"""
Guided backpropagation applied on the cod otolith model presented at NORDSTAT 2021.

Created on Wed Jun 16 11:36:32 2021

@author: iverm
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from modules.imageloader import load_images
from modules.analysis.guided_backpropagation import build_gb_model_seq as build_gb_model
from modules.analysis.utils import compute_gradients


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
    
    _outputs = Dense(1, activation='sigmoid')(x)
        
    _model = tf.keras.Model(inputs=inputs, outputs=_outputs)
    
    _model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    return _model


path = r'C:\Users\iverm\Google Drive\Masteroppgave\Data\Torskeotolitter\standard'

k = 5
sets = load_images(path, (128, 128), 5, seed=123)

# use first three sets for training
x_tr = np.concatenate([generator['images'] for generator in [sets[i] for i in [0, 1, 4]]])
y_tr = np.concatenate([generator['labels'] for generator in [sets[i] for i in [0, 1, 4]]])

# use fourth set for validation
x_va = sets[2]['images']
y_va = sets[2]['labels']

# use fifth set for testing
x_te = sets[3]['images']
y_te = sets[3]['labels']

# fit model using early stopping on validation data
model = build_model()

callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]

history = model.fit(
    x_tr,
    y_tr,
    epochs=100,
    validation_data=(x_va, y_va),
    callbacks=callbacks)


# build gb_model for NEAC (class 1) which replaces ReLU activations in model with modified ReLU's
gb_model_neac = build_gb_model(model)

# build gb_model for NCC (class 0) by inverting the output probability
gb_model_ncc = tf.keras.Model(
    inputs=gb_model_neac.inputs,
    outputs=[1-output for output in gb_model_neac.outputs]
)

# store test data classified as NEAC and NCC in two separate variables
# and compute guided backpropagation relevance for both variables
pred = model.predict(x_te).round().flatten()
neac = tf.convert_to_tensor(x_te[np.where(pred == 1)])
ncc = tf.convert_to_tensor(x_te[np.where(pred == 0)])
grads_neac = compute_gradients(neac, gb_model_neac, 0)
grads_ncc = compute_gradients(ncc, gb_model_ncc, 0)

# display the 1st heatmap of each stock
_, ax = plt.subplots(1, 2)

ax[0].imshow(grads_neac[0], cm.jet)
ax[0].axis('off')
ax[1].imshow(grads_ncc[0], cm.jet)
ax[1].axis('off')

plt.show()

# select an image for further analysis
tensor = tf.convert_to_tensor(x_te[0])

# define models that returns the output of the 1st, 2nd and 3rd conv block
layer_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[5].output)
layer_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[9].output)
layer_3 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[13].output)

layer1_output = np.mean(layer_1(tensor), axis=0)
layer2_output = np.mean(layer_2(tensor), axis=0)
layer3_output = np.mean(layer_3(tensor), axis=0)

# organize figure in a 3 by 32 grid
fig = plt.figure()
gridspec.GridSpec(3, 32)

# display the first 16 hidden feature maps of layer 2 in row 1
for i in range(16):
    plt.subplot2grid((3, 32), (0, i*2), colspan=2, rowspan=1)
    plt.imshow(layer2_output[:, :, i], cm.afmhot)
    plt.axis('off')

# display the last 16 hidden feature maps of layer 2 in row 2
for i in range(16):
    plt.subplot2grid((3, 32), (1, i*2), colspan=2, rowspan=1)
    plt.imshow(layer2_output[:, :, 16+i], cm.afmhot)
    plt.axis('off')

# display the 32 hidden feature maps of layer 3 in row 3
for i in range(32):
    plt.subplot2grid((3, 32), (2, i), colspan=1, rowspan=1)
    plt.imshow(layer3_output[:, :, i], cm.afmhot)
    plt.axis('off')
    