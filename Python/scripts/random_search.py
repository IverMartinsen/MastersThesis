"""
Requires tensorflow 2.5.0, numpy 1.19.5, keras-tuner, pillow, pandas and matplotlib
"""
import sys
import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
from modules.utils import stratified_idxs

# Load images and features
images = np.load(
    r'C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith '
    r'images\Data\Blåkveiteotolitter\images256.npy')
dataframe = pd.read_csv(
    r'C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith '
    r'images\Data\Blåkveiteotolitter\dataframe.csv')

# Data with unknown sex
img_unknown = images[np.where(dataframe['sex'] == 'unknown')]
lab_unknown = np.asarray(dataframe['age'].iloc[np.where(dataframe['sex'] == 'unknown')])
sex_unknown = np.asarray(dataframe['sex'].iloc[np.where(dataframe['sex'] == 'unknown')])

# Data with known sex
img_known = images[np.where(dataframe['sex'] != 'unknown')]
lab_known = np.asarray(dataframe['age'].iloc[np.where(dataframe['sex'] != 'unknown')])
sex_known = np.asarray(dataframe['sex'].iloc[np.where(dataframe['sex'] != 'unknown')])

# Stratify the data and create tf.Datasets for training and validation
# We use images with unknown sex only for training
# We use 20 % of the images with known sex for testing
# We use batch size 32
train_idx, valid_idx = stratified_idxs(lab_known, splits=(0.8, 0.2), seed=1234)

batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((
    (np.concatenate((img_unknown, img_known[train_idx]), axis=0), np.concatenate((sex_unknown, sex_known[train_idx]))),
    np.concatenate((lab_unknown, lab_known[train_idx]))
)).shuffle(buffer_size=4000).batch(batch_size=batch_size).take(2)

valid_ds = tf.data.Dataset.from_tensor_slices((
    (img_known[valid_idx], sex_known[valid_idx]), lab_known[valid_idx]
)).batch(batch_size=batch_size).take(2)

# Compute mean values to use as bias initial values
mean_values = (
    0,
    0,
    np.mean(dataframe['age'].iloc[np.where(dataframe['sex'] == 'female')]),
    np.mean(dataframe['age'].iloc[np.where(dataframe['sex'] == 'male')]),
    np.mean(dataframe['age'].iloc[np.where(dataframe['sex'] == 'unknown')]))

# Define a build model function
def build_model(hp):
    """

    Parameters
    ----------
    hp

    Returns
    -------
    tf.Model
    """
    img_input = tf.keras.layers.Input(img_known.shape[1:4])
    cat_input = tf.keras.layers.Input(shape=(1,), name='gender', dtype='string')

    # Create layer for mapping categorical labels to int
    # Fit index layer on training data
    # Create layer for one-hot-encoding the categorical labels
    index_layer = tf.keras.layers.experimental.preprocessing.StringLookup()
    index_layer.adapt(tf.constant(dataframe['sex']))
    encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        num_tokens=index_layer.vocabulary_size(), output_mode='binary')

    # Define pretrained base model without classification head. Use global average pooling on output.
    base_model = tf.keras.applications.xception.Xception(input_shape=img_known.shape[1:4], include_top=False,
                                                         pooling='avg')

    # First we process the images
    x = tf.keras.applications.xception.preprocess_input(img_input)
    x = tf.keras.layers.experimental.preprocessing.RandomTranslation(0, 0.1)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='constant')(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(hp.Float("dropout", 0, 0.5, 0.1))(x)
    x = tf.keras.layers.Dense(5, 'relu', bias_initializer=tf.keras.initializers.Constant(mean_values))(x)
    # Then we us multiplication to get the gender conditional age predictions
    outputs = tf.keras.layers.Dot(axes=1)([x, encoding_layer(index_layer(cat_input))])
    # Finally, we concatenate the age prediction with the one-hot sex matrix
    _model = tf.keras.models.Model([img_input, cat_input], outputs)

    # Compile model using custom loss function
    _model.compile(
        tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-6, 1e-1, sampling="log")),
        tf.keras.losses.MeanSquaredError()
    )

    return _model


# Define parameter tuner instance for random search
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=20,
    project_name='_random_search_')

# Execute random search
tuner.search(
    train_ds,
    epochs=100,
    validation_data=valid_ds,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])

# Save summary of tuner
original_stdout = sys.stdout
with open('hyperparameters.txt', 'w') as file:
    sys.stdout = file
    tuner.results_summary()
    sys.stdout = original_stdout
