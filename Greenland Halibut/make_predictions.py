import tensorflow as tf

path_to_model = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Prosjekter/Deep learning applied to fish otolith images/src/saved_models/model1"

custom_objects = {"index_layer": None}

model = tf.keras.models.load_model(path_to_model)

model.compile
