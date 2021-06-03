from tensorflow.keras import Model
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from dataloader import dataloader
'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\raw'


#data2 = Dataset((128, 128), keep_aspect=True)
#data2.load(r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\unknown')
#sets2, _ = data2.kfoldsplit(1)

train_ds, valid_ds, test_ds = dataloader(path, (128, 128), 1, [0.6, 0.2, 0.2])
train_ds = train_ds.shuffle(1000)



from model import CodNet as model

model.load_weights(r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Model Checkpoints\raw_weights\trained')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

layer_outputs = [layer.output for layer in [model.layers[i] for i in [6, 10, 14, 18]]]

extractor = Model(inputs = model.input, outputs = layer_outputs)

img_tensor = list(test_ds)[0][0][2:3]

features = extractor.predict(img_tensor)


fig, axes = plt.subplots(2, 4)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(features[0][0, :, :, i], 'gray')
    ax.axis('off')

plt.imshow(img_tensor[0], 'gray')


model(img_tensor)


model.output


with tf.GradientTape() as tape:
    # get final convolutional layer
    tape.watch(img_tensor)
    last_conv_layer = model.get_layer('max_pooling2d_7') # conv2d_7
    iterate = tf.keras.Model([model.inputs],
                             [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(img_tensor)
    class_out = model_out[:, ]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
heatmap = tf.reduce_mean(heatmap, axis = 0)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((8, 8))
plt.matshow(heatmap)
plt.show()



INTENSITY = 0.5

heatmap = cv2.resize(heatmap, (img_tensor.shape[2], img_tensor.shape[1]))
heatmap /= np.max(heatmap)

img = rgb2gray(heatmap) * INTENSITY + np.mean(img_tensor.numpy(), axis = 0).reshape(128, 128)/255.
img = cv2.addWeighted(rgb2gray(heatmap), 0.7, np.mean(img_tensor.numpy(), axis = 0).reshape(128, 128), 0.3, 0)
img /= np.max(img)

img = heatmap * INTENSITY + img_tensor[0].numpy().reshape(128, 128)/255.0


#cv2.imshow('', img)
fig, ax1 = plt.subplots()
ax1.imshow(img_tensor[0], 'gray', alpha = 1)
ax1.imshow(heatmap, 'jet', alpha = 0.4)
ax1.axis('off')

plt.imshow(img, 'jet')

test = np.where(list(test_ds)[0][1].numpy() == 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])