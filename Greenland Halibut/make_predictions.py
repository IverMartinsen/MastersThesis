# Script for predicting otolith age
#
# Must be accompanied by the following
#
#   1) A folder named 'saved_models' with saved TensorFlow models
#   2) A folder named 'images_to_predict' with images to predict
#
# Run through command line by 
# python make_predictions.py --sex female
#
# where female may be replaced by 'male' or 'unknown'

import logging
logging.getLogger('tensorflow').disabled = True
import os
import warnings
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf


required_image_size = (256, 256)

parser = argparse.ArgumentParser("make_predictions.py")

parser.add_argument("-s", "--sex", type=str, action="store")

if __name__ == "__main__":

    path_to_models = "saved_models"
    models = os.listdir(path_to_models)
    models = [tf.keras.models.load_model(os.path.join(path_to_models, model)) for model in models]

    path_to_images = "images_to_predict"
    images = os.listdir(path_to_images)

    sex = parser.parse_args().sex
    if sex == "female":
        factor = np.array((1, 0))
    elif sex == "male":
        factor = np.array((0, 1))
    elif sex == "unknown" or sex is None:
        warnings.warn("Sex not provided, estimating based on mean.")
        factor = np.array((0.5, 0.5))

    for image in images:
        filename = image
        image_path = os.path.join(path_to_images, image)
        image = np.array(Image.open(image_path))[None, :, :, :]
        image_size = image.shape[1:3]
        
        if image_size != required_image_size:
            warnings.warn(f"Input not of required size. Reshaping image to {required_image_size}.")
            image = np.array(Image.open(image_path).resize(required_image_size))[None, :, :, :]
        
        predictions = np.zeros(len(models))
        
        for i, model in enumerate(models):
            predictions[i] = np.sum(np.array(model.predict(image)[0][1:3])*factor)
        
        age_mean = np.round(np.mean(predictions)).astype(int)
        age_std = np.round(np.std(predictions), 2)
        
        print(f"Prediction for {filename}")
        print(f"Sex: {sex}")
        print(f"Age: {age_mean} years")
        print(f"Standard deviation: {age_std} years")