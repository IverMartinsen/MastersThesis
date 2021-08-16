# -*- coding: utf-8 -*-
"""
An example of applying transfer learning on otolith images.

Created on Mon Jun 14 19:48:11 2021

@author: iverm
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.imageloader import imageloader
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


'''
load images
'''
path = r'C:\Users\iverm\Google Drive\Masteroppgave\Data\Torskeotolitter\standard'
train_ds, valid_ds, test_ds = imageloader(
    path, (128, 128), splits=(0.6, 0.2, 0.2), seed=123, mode='RGB')


'''
display training images
'''
images = train_ds['images'][:9]
labels = train_ds['labels'][:9]

plt.figure(figsize=(10, 10))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(train_ds.class_names[labels[i]])
    plt.axis("off")

        
'''
load MobileNet
'''
# Create the base model from the pre-trained model MobileNet V2
IMG_SIZE = (128, 128)
IMG_SHAPE = IMG_SIZE + (3,)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 140

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


'''
define model
'''
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

base_learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()


'''
fit model
'''
BATCH_SIZE = 32
initial_epochs = 100

x_tr = train_ds['images']
y_tr = train_ds['labels']
x_va = valid_ds['images']
y_va = valid_ds['labels']

callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]


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
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
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
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'],
)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]


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
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()















from modules.pixelanalysis import generate_path_inputs, integrate_grads, compute_grads
from modules.image_tools import normalize
import numpy as np

input_img = test_ds['images'][5]

baseline_img = np.zeros_like(input_img)
path_imgs = generate_path_inputs(baseline_img, input_img, 50)
grads = compute_grads(path_imgs, model, 0) / 255
ig_grads = integrate_grads(grads)

plt.imshow(np.abs(np.sum(ig_grads, axis = 2)), cmap=plt.cm.inferno)
plt.imshow(input_img[:, :, 0]/255, 'gray', alpha=0.4)

-np.log(1 / model(baseline_img[np.newaxis, :, :, :]) - 1)
-np.log(1 / model(input_img[np.newaxis, :, :, :]) - 1)

model(baseline_img[np.newaxis, :, :, :])

test = np.random.uniform(size=input_img.shape) * 127.5
test = np.ones(shape = input_img.shape) * 127.5
test = np.mean(test_ds['images'][np.where(test_ds['labels'] == 0)], axis = 0)
model(test[np.newaxis, :, :, :])
model(input_img[np.newaxis, :, :, :])
