import sys

sys.path.append('/home/iver')

import pandas as pd
import tensorflow as tf
import numpy as np
from stratified_idxs import stratified_idxs

# Load dataframe of features
df = pd.read_csv('/datasets/dataframe.csv')

# Locate data points with complete set of features
notna = np.all(np.array(df.notna()), axis = 1)

# Drop data with incomplete set of features
df = df.dropna()

# Only use images with complete set of features
images = np.load('/datasets/images128.npy')[notna]

# Create stratified indices for selecting datasets for training etc.
strata_idxs = stratified_idxs(df['age'], 10, seed=123)

# Define utility functions for creating tensorflow compatible datasets
def set_from_idx(idx, training=False):
    if training:
        return tf.data.Dataset.from_tensor_slices(
            ((tf.convert_to_tensor(df['sex'].iloc[idx]), images[idx]), df['age'].iloc[idx])).shuffle(len(idx)).batch(batch_size)
    else:
        return tf.data.Dataset.from_tensor_slices(
            ((tf.convert_to_tensor(df['sex'].iloc[idx]), images[idx]), df['age'].iloc[idx])).batch(batch_size)

def mat_from_idx(idx, y=None):
    if y is None:
        return tf.stack((tf.constant(1, shape=len(idx)) ,df['length'].iloc[idx], df['length'].iloc[idx]*(df['sex'].iloc[idx] == 'male')), axis=1)
    else:
        return tf.stack((tf.constant(1, shape=len(idx), df['length'].iloc[idx], df['length'].iloc[idx]*(df['sex'].iloc[idx] == 'male'), y), axis=1)


def build_model():
    # Create layer for mapping categorical labels to int
    Index = tf.keras.layers.StringLookup()
    # Fit index layer on training data
    Index.adapt(tf.constant(df['sex']))

    # Create layer for one-hot-encoding the categorical labels
    Encoding = tf.keras.layers.CategoryEncoding(num_tokens=Index.vocabulary_size(), output_mode='one_hot')

    # Define pretrained base model without classification head. Use global average pooling on output.
    base_model = tf.keras.applications.Xception(
        input_shape=image_size + (3,),
        include_top=False,
        pooling='avg')

    # Define full model. Note that by setting training=False in the base model
    # we always run the model in inference mode.
    img_input = tf.keras.layers.Input(image_size + (3,))
    cat_input = tf.keras.Input(shape=(1,), name='gender', dtype='string')

    gender = Encoding(Index(cat_input))

    # First we process the images
    x = tf.keras.applications.xception.preprocess_input(img_input)
    x = tf.keras.layers.RandomTranslation(0, 0.1)(x)
    x = tf.keras.layers.RandomRotation(0.1, fill_mode='constant')(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(3, 'relu')(x)
    # Then we us multiplication to get the gender conditional age predictions
    outputs = tf.keras.layers.Dot(axes=1)([x, gender])
    # Finally we concatenate the age prediction with the one-hot sex matrix
    model = tf.keras.models.Model([cat_input, img_input], outputs)

    # Compile model using custom loss function
    model.compile(tf.keras.optimizers.Adam(0.00046625), tf.keras.losses.MeanSquaredError())

    return model


def build_linear_model(num_predictors):
    inputs = tf.keras.layers.Input(shape=(num_predictors,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    linear_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    linear_model.compile(tf.keras.optimizers.Adam(1e-3), tf.keras.losses.MeanSquaredError())

    return linear_model

image_size = images.shape[1:3]
max_epochs = 1
batch_size = 32
model = None
linear_model = None
patience = 1

summary = pd.DataFrame(index=range(10), columns=['loss1', 'loss2', 'loss3'])
results = pd.DataFrame()

for i in range(len(strata_idxs)):

    test_idx = strata_idxs[i]

    if i + 1 < len(strata_idxs):
        valid_idx = strata_idxs[i + 1]
        train_idx = np.concatenate(np.delete(strata_idxs, [i, i + 1]))
    else:
        valid_idx = strata_idxs[0]
        train_idx = np.concatenate(np.delete(strata_idxs, [i, 0]))

    tf.keras.backend.clear_session()

    del model
    del linear_model

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

    y1 = model.predict(set_from_idx(test_idx)).flatten()

    summary['loss1'].iloc[i] = model.evaluate(set_from_idx(test_idx))

    # Predict age by length
    X = mat_from_idx(np.concatenate((train_idx, valid_idx)))
    y = df['age'].iloc[np.concatenate((train_idx, valid_idx))]

    y2 = X @ np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    summary['loss2'].iloc[i] = tf.keras.losses.mean_absolute_error(mat_from_idx(test_idx), df['age'].iloc[test_idx])

    # Predict age by combining models
    linear_model = build_linear_model(num_predictors=3)

    callbacks = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)

    linear_model.fit(
        mat_from_idx(train_idx, y=model.predict(set_from_idx(train_idx)).flatten()),
        df['age'].iloc[train_idx],
        epochs=max_epochs,
        validation_data=(
        mat_from_idx(valid_idx, y=model.predict(set_from_idx(valid_idx)).flatten()), df['age'].iloc[valid_idx]),
        callbacks=callbacks
    )

    y3 = linear_model.predict(mat_from_idx(test_idx, y=y1)).flatten()
    summary['loss3'].iloc[i] = linear_model.evaluate(mat_from_idx(test_idx, y=y1), df['age'].iloc[test_idx])

    # Store results
    result = pd.DataFrame(
        {'filename': df['filename'].iloc[test_idx],
         'age': df['age'].iloc[test_idx],
         'length': df['length'].iloc[test_idx],
         'sex': df['sex'].iloc[test_idx],
         'y1': y1,
         'y2': y2,
         'y3': y3}
    )

    result.to_csv('/home/iver/results/trial_' + str(i+1) + '.csv')
    summary.to_csv('/home/iver/results/summary_' + str(i+1) + '.csv')

    if i == 0:
        results = result
    else:
        results = pd.merge(results, result, how='outer')

# Save results to file
summary.to_csv('/home/iver/results/summary.csv')
results.to_csv('/home/iver/results/results.csv')


