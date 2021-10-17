import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

file_path = r'C:\Users\iverm\Desktop\UiT\Data\Grønlandskveiteotolitter\greenland_halibut_std'

image_size = (299, 299)

# load images from directory in alphabetical order
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    file_path,
    label_mode=None,
    image_size=image_size,
    shuffle=False)

# stack images into numpy array
images = np.stack(list(dataset.unbatch().as_numpy_iterator()))

np.save(r'C:\Users\iverm\Desktop\UiT\Data\images299.npy', images)

plt.imshow(images[0] / 255.0)

np.load(r'C:\Users\iverm\Google Drive\images299.npy')
