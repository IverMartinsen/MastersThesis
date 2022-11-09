"""
Script for running trial procedure on external server.
Requires tensorflow 2.5.0, numpy 1.19.5, keras-tuner, pillow, pandas and matplotlib.
"""
import wandb
import pandas as pd
import tensorflow as tf
import numpy as np
from utils import stratified_idxs
from models import get_full_model, get_deployable_model, get_categorical_model

save_results_to_csv = False
save_results_using_wandb = True

config = {
    "max_epochs": 100,
    "batch_size": 32,
    "patience": 20,
    "learning_rate": 1e-4,
    "dropout_rate": 0.4,
}

df = pd.read_csv("data/dataframe.csv")

path_to_images = "data/images256.npy"

img_known_sex = np.load(path_to_images)[np.where(df["sex"] != "unknown")]
img_unknown_sex = np.load(path_to_images)[np.where(df["sex"] == "unknown")]

# Create stratified indices for selecting datasets for training etc.
strata_idxs = stratified_idxs(
    df["age"].iloc[np.where(df["sex"] != "unknown")], 10, seed=1234
)


# Define utility functions for creating tensorflow compatible datasets from array of indices
def test_set_from_idx(idx):

    return tf.data.Dataset.from_tensor_slices(
        (
            (
                tf.convert_to_tensor(
                    df["sex"].iloc[np.where(df["sex"] != "unknown")].iloc[idx]
                ),
                img_known_sex[idx],
            ),
            df["age"].iloc[np.where(df["sex"] != "unknown")].iloc[idx],
        )
    ).batch(config["batch_size"])


def train_set_from_idx(idx):

    return (
        tf.data.Dataset.from_tensor_slices(
            (
                (
                    tf.convert_to_tensor(
                        np.concatenate(
                            (
                                df["sex"].iloc[np.where(df["sex"] == "unknown")],
                                df["sex"]
                                .iloc[np.where(df["sex"] != "unknown")]
                                .iloc[idx],
                            )
                        )
                    ),
                    np.concatenate((img_unknown_sex, img_known_sex[idx])),
                ),
                np.concatenate(
                    (
                        df["age"].iloc[np.where(df["sex"] == "unknown")],
                        df["age"].iloc[np.where(df["sex"] != "unknown")].iloc[idx],
                    )
                ),
            )
        )
        .shuffle(len(idx))
        .batch(config["batch_size"])
    )


def mat_from_idx(idx):
    # Create matrix
    return tf.stack(
        (
            tf.constant(1, shape=len(idx)),
            df["length"].iloc[np.where(df["sex"] != "unknown")].iloc[idx],
            df["length"].iloc[np.where(df["sex"] != "unknown")].iloc[idx]
            * (df["sex"].iloc[np.where(df["sex"] != "unknown")].iloc[idx] == "male"),
        ),
        axis=1,
    )


image_shape = img_known_sex.shape[1:4]
model = None

# Create dataframes for storing summary (per training session) results and individual (per image) results
summary = pd.DataFrame(
    index=range(10), columns=["Test loss (DL)", "Test loss (regression)"]
)
results = pd.DataFrame()

# Compute mean values to use as bias initial values
mean_values = (
    0,
    np.mean(df["age"].iloc[np.where(df["sex"] == "female")]),
    np.mean(df["age"].iloc[np.where(df["sex"] == "male")]),
    np.mean(df["age"].iloc[np.where(df["sex"] == "unknown")]),
)

vocabulary = np.unique(df["sex"])

for i in range(len(strata_idxs)):

    if save_results_using_wandb:
        run = wandb.init(
            project="Greenland halibut cross validation", reinit=True, config=config
        )

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

    print(f"\nStarting trial {i + 1}\n")

    deployable_model = get_deployable_model(
        image_shape=image_shape,
        initial_bias=mean_values,
        dropout_rate=config["dropout_rate"],
    )
    categorical_model = get_categorical_model(vocabulary=vocabulary)
    model = get_full_model(deployable_model, categorical_model, image_shape)

    model.compile(
        tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        tf.keras.losses.MeanSquaredError(),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config["patience"], restore_best_weights=True
        )
    ]
    if save_results_using_wandb:
        callbacks.append(wandb.keras.WandbCallback())

    model.fit(
        train_set_from_idx(train_idx),
        epochs=config["max_epochs"],
        validation_data=test_set_from_idx(valid_idx),
        callbacks=callbacks,
    )

    # Compute deep learning predictions
    y1 = model.predict(test_set_from_idx(test_idx)).flatten()

    # Store test loss in dataframe
    summary["Test loss (DL)"].iloc[i] = model.evaluate(test_set_from_idx(test_idx))

    # Predict age of test data using length
    X = mat_from_idx(np.concatenate((train_idx, valid_idx)))
    y = (
        df["age"]
        .iloc[np.where(df["sex"] != "unknown")]
        .iloc[np.concatenate((train_idx, valid_idx))]
    )
    w = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    # Compute linear regression predictions
    y2 = np.matmul(mat_from_idx(test_idx), w)

    # Store test loss in dataframe
    summary["Test loss (regression)"].iloc[i] = tf.keras.losses.mean_squared_error(
        y2,
        np.asarray(
            df["age"].iloc[np.where(df["sex"] != "unknown")].iloc[test_idx]
        ).astype("float"),
    ).numpy()

    # Store result of current trial in dataframe
    result = pd.DataFrame(
        {
            "filename": df["filename"]
            .iloc[np.where(df["sex"] != "unknown")]
            .iloc[test_idx],
            "age": df["age"].iloc[np.where(df["sex"] != "unknown")].iloc[test_idx],
            "length": df["length"]
            .iloc[np.where(df["sex"] != "unknown")]
            .iloc[test_idx],
            "sex": df["sex"].iloc[np.where(df["sex"] != "unknown")].iloc[test_idx],
            "y1": y1,
            "y2": y2,
        }
    )

    if save_results_using_wandb:
        run.log(
            {
                "Test summary": wandb.Table(dataframe=summary),
                "Test results": wandb.Table(dataframe=result),
            }
        )

    # Save result of current trial to file, just in case
    if save_results_to_csv:
        result.to_csv("trial_" + str(i + 1) + ".csv")
        summary.to_csv("summary_" + str(i + 1) + ".csv")

    # Merge existing results with the result from current trial
    if i == 0:
        results = result
    else:
        results = pd.concat((results, result))

    deployable_model.save(f"saved_models/model{i+1}", save_format="tf")

    if save_results_using_wandb:
        run.finish()

# Save results to files
if save_results_to_csv:
    summary.to_csv("summary.csv")
    results.to_csv("results.csv")
