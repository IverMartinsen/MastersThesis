import glob
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import savemat

required_image_size = 256, 256 # Model input must be 256, 256
 
# Load models from path
path_to_models = "./saved_models"
models = sorted(os.listdir(path_to_models))
models = [tf.keras.models.load_model(os.path.join(path_to_models, model)) for model in models]

# List, sort, load and resize images from path
path_to_images = "./images_to_predict"
images = sorted(glob.glob(path_to_images + "/*.jpg"))

for j, image_path in enumerate(images):
    image = np.array(Image.open(image_path)) # Load image
    
    if image.shape[1:3] != required_image_size: # Resize image if necessary
        image = np.array(Image.open(image_path).resize(required_image_size))
    
    images[j] = image / 255. # Replace image path with image array

images = tf.convert_to_tensor(images) # Convert to tensor

# Store gradients in a multidimensional array of shape (#models, #sex, #images, height, width, channels)
gradients = np.zeros((len(models), 3, len(images), 256, 256, 3))

# Iterate models, sex and images
for i, model in enumerate(models):
    print(f"Calculating gradients for model {i + 1} of {len(models)}")
    for j in range(3):
        print(f"---Sex {j + 1} of 3---")
        for k in range(len(images)):
            print(f"------Image {k + 1} of {len(images)}------")
            image = images[k][None, :, :, :]
            with tf.GradientTape() as tape:
                tape.watch(image)
                outputs = model(image, training=False)
                outputs = tf.slice(outputs, [0, j + 1], [1, 1])
            grads = tape.gradient(outputs, image)
            gradients[i, j, k, :, :, :] = grads.numpy()
    print("Done!")

mdic = {'a': gradients}
 
savemat("./my_gradients.mat", mdic)
