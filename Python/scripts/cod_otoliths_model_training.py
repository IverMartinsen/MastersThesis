# -*- coding: utf-8 -*-
"""
Script showing how to fit a deep learning model to cod otolith images
by applying transfer learning.

Created on Mon Jun 14 19:48:11 2021

@author: iverm
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.imageloader import load_images
from modules.image import normalize
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications.xception import preprocess_input
from modules.analysis.guided_backpropagation import build_gb_model_nonseq


'''
load images
'''
path = r'C:\Users\iverm\Google Drive\Data\Torskeotolitter\standard'
train_ds, valid_ds, test_ds = load_images(path, (128, 128), splits=(0.6, 0.2, 0.2), seed=123, mode='RGB')


'''
display training images
'''
images = train_ds['images'][:9]
labels = train_ds['labels'][:9]

plt.figure(figsize=(10, 10))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(normalize(images[i]), 'gray')
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")

        
'''
load MobileNet
'''
# Create the base model from the pre-trained Xception model
IMG_SIZE = (128, 128)
IMG_SHAPE = IMG_SIZE + (3,)

base_model = tf.keras.applications.Xception(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet')

# Freeze all layers in base model
for layer in base_model.layers:
    layer.trainable = False


'''
define model
'''
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

base_learning_rate = 1e-3

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])


'''
fit model
'''
BATCH_SIZE = 32
initial_epochs = 100

x_tr = train_ds['images']
y_tr = train_ds['labels']
x_va = valid_ds['images']
y_va = valid_ds['labels']

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]


history = model.fit(x_tr,
                    y_tr,
                    batch_size=BATCH_SIZE,
                    epochs=initial_epochs,
                    validation_data=(x_va, y_va),
                    callbacks=callbacks)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


'''
fine tune model
'''
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

history_fine = model.fit(x_tr,
                         y_tr,
                         batch_size=BATCH_SIZE,
                         epochs=100,
                         validation_data=(x_va, y_va),
                         callbacks=callbacks)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


'''
Save trained model
'''
model.save(r'C:\Users\iverm\Google Drive\Artikkel om torskeotolitter\Saved models\MobileNetV2_' + tf.__version__)


'''
Build Guided Backprop model and save
'''
base_model = tf.keras.applications.Xception(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet')

gb_base_model = build_gb_model_nonseq(base_model, instance=tf.keras.layers.ReLU)

inputs = tf.keras.Input(IMG_SHAPE)
x = preprocess_input(inputs)
x = gb_base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(3, 'softmax')(x)

gb_model = tf.keras.Model(inputs, outputs)

gb_model.compile(    
    tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics='accuracy')

gb_model.set_weights(model.get_weights())

gb_model.save(r'C:\Users\iverm\Google Drive\Artikkel om torskeotolitter\Saved models\MobileNetV2_gb_' + tf.__version__)
