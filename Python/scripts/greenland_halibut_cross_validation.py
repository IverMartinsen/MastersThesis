"""
Script for running trial procedure on external server.
Requires tensorflow 2.5.0, numpy 1.19.5, keras-tuner, pillow, pandas and matplotlib.
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from modules.utils import stratified_idxs

# Load dataframe of features
df = pd.read_csv(r'C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith '
                 r'images\Data\Blåkveiteotolitter\dataframe.csv')

# Load images
img_known = np.load(r'C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith '
                    r'images\Data\Blåkveiteotolitter\images128.npy')[np.where(df['sex'] != 'unknown')]
img_unknown = np.load(r'C:\Users\UiT\OneDrive - UiT Office 365\Desktop\Deep learning applied to fish otolith '
                      r'images\Data\Blåkveiteotolitter\images128.npy')[np.where(df['sex'] == 'unknown')]

# Create stratified indices for selecting datasets for training etc.
strata_idxs = stratified_idxs(df['age'].iloc[np.where(df['sex'] != 'unknown')], 10, seed=1234)


# Define utility functions for creating tensorflow compatible datasets from array of indices
def set_from_idx(idx, training=False):
    # If training, also include images with unknown sexes
    if training:
        return tf.data.Dataset.from_tensor_slices((
            (tf.convert_to_tensor(np.concatenate((
                df['sex'].iloc[np.where(df['sex'] == 'unknown')],
                df['sex'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx]
            ))), np.concatenate((img_unknown, img_known[idx]))),
            np.concatenate((
                df['age'].iloc[np.where(df['sex'] == 'unknown')],
                df['age'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx])))).shuffle(len(idx)).batch(batch_size)
    # Else, if testing or validation, only include images with complete features
    else:
        return tf.data.Dataset.from_tensor_slices((
            (tf.convert_to_tensor(df['sex'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx]), img_known[idx]),
            df['age'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx])).batch(batch_size)


def mat_from_idx(idx):
    # Create matrix
    return tf.stack((
        tf.constant(1, shape=len(idx)),
        df['length'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx],
        df['length'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx] *
        (df['sex'].iloc[np.where(df['sex'] != 'unknown')].iloc[idx] == 'male')), axis=1)


def build_model():
    # Create layer for mapping categorical labels to int
    index_layer = tf.keras.layers.experimental.preprocessing.StringLookup()

    # Fit index layer on training data
    index_layer.adapt(tf.constant(df['sex']))

    # Create layer for one-hot-encoding the categorical labels
    encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        num_tokens=index_layer.vocabulary_size(), output_mode='binary')

    # Define pretrained base model without classification head. Use global average pooling on output.
    base_model = tf.keras.applications.Xception(
        input_shape=image_shape, include_top=False, pooling='avg')

    # Define full model. Note that by setting training=False in the base model
    # we always run the model in inference mode.
    img_input = tf.keras.layers.Input(image_shape)
    cat_input = tf.keras.Input(shape=(1,), name='gender', dtype='string')

    # First we process the images
    x = tf.keras.applications.xception.preprocess_input(img_input)
    x = tf.keras.layers.experimental.preprocessing.RandomTranslation(0, 0.1)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='constant')(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(5, 'relu', bias_initializer=tf.keras.initializers.Constant(mean_values))(x)
    # Then we us multiplication to get the gender conditional age predictions
    outputs = tf.keras.layers.Dot(axes=1)([x, encoding_layer(index_layer(cat_input))])
    # Finally, we concatenate the age prediction with the one-hot sex matrix
    _model = tf.keras.models.Model([cat_input, img_input], outputs)

    # Compile model using custom loss function
    _model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate), tf.keras.losses.MeanSquaredError())

    return _model


# Set hyperparameters for training
image_shape = img_known.shape[1:4]
max_epochs = 1
batch_size = 32
patience = 20
learning_rate = 1e-4
dropout_rate = 0.4
model = None

# Create dataframes for storing summary (per training session) results and individual (per image) results
summary = pd.DataFrame(index=range(10), columns=['loss1', 'loss2'])
results = pd.DataFrame()

# Compute mean values to use as bias initial values
mean_values = (
    0,
    0,
    np.mean(df['age'].iloc[np.where(df['sex'] == 'female')]),
    np.mean(df['age'].iloc[np.where(df['sex'] == 'male')]),
    np.mean(df['age'].iloc[np.where(df['sex'] == 'unknown')]))

# Iterate through the stratified indices to create training, validation and test sets
for i in range(len(strata_idxs)):

    # choose strata i for testing
    test_idx = strata_idxs[i]

    # If possible, choose next strata for validation and the rest for training
    if i + 1 < len(strata_idxs):
        valid_idx = strata_idxs[i + 1]
        train_idx = np.concatenate(np.delete(strata_idxs, [i, i + 1]))
    # Otherwise, choose first one for validation and the rest for training
    else:
        valid_idx = strata_idxs[0]
        train_idx = np.concatenate(np.delete(strata_idxs, [i, 0]))

    tf.keras.backend.clear_session()

    del model

    print(f'\nStarting trial {i + 1}\n')

    # Predict age by image
    model = build_model()

    callbacks = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

    model.fit(
        set_from_idx(train_idx, training=True),
        epochs=max_epochs,
        validation_data=set_from_idx(valid_idx),
        callbacks=callbacks
    )

    # Compute deep learning predictions
    y1 = model.predict(set_from_idx(test_idx)).flatten()

    # Store test loss in dataframe
    summary['loss1'].iloc[i] = model.evaluate(set_from_idx(test_idx))

    # Predict age of test data using length
    X = mat_from_idx(np.concatenate((train_idx, valid_idx)))
    y = df['age'].iloc[np.where(df['sex'] != 'unknown')].iloc[np.concatenate((train_idx, valid_idx))]
    w = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    # Compute linear regression predictions
    y2 = np.matmul(mat_from_idx(test_idx), w)

    # Store test loss in dataframe
    summary['loss2'].iloc[i] = tf.keras.losses.mean_squared_error(
        y2, np.asarray(df['age'].iloc[np.where(df['sex'] != 'unknown')].iloc[test_idx]).astype('float')).numpy()

    # Store result of current trial in dataframe
    result = pd.DataFrame({
        'filename': df['filename'].iloc[np.where(df['sex'] != 'unknown')].iloc[test_idx],
        'age': df['age'].iloc[np.where(df['sex'] != 'unknown')].iloc[test_idx],
        'length': df['length'].iloc[np.where(df['sex'] != 'unknown')].iloc[test_idx],
        'sex': df['sex'].iloc[np.where(df['sex'] != 'unknown')].iloc[test_idx],
        'y1': y1,
        'y2': y2})

    # Save result of current trial to file, just in case
    result.to_csv('trial_' + str(i + 1) + '.csv')
    summary.to_csv('summary_' + str(i + 1) + '.csv')

    # Merge existing results with the result from current trial
    if i == 0:
        results = result
    else:
        results = pd.concat((results, result))

# Save results to files
summary.to_csv('summary.csv')
results.to_csv('results.csv')
