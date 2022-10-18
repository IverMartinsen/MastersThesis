"""
Script for running trial procedure for the prediction of
Greenland halibut age using linear regression on length.
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from modules.stratified_idxs import stratified_idxs

# Load dataframe of features
df = pd.read_csv(r'C:\Users\iverm\OneDrive - UiT Office 365\UiT\Data\Grønlandskveiteotolitter\dataframe.csv')

# Drop data with incomplete set of features
df = df.dropna()

# Create stratified indices for 10 subsets
strata_idxs = stratified_idxs(df['age'], 10, seed=123)


# Define utility function for creating tensorflow compatible datasets from an array of indices
def mat_from_idx(idx):
    return tf.stack(
        (tf.constant(1, shape=len(idx)),
         df['length'].iloc[idx],
         df['length'].iloc[idx]*(df['sex'].iloc[idx] == 'male')),
        axis=1)


# Create dataframes for storing summary loss for each trial, in addition to individual results
summary = pd.DataFrame(index=range(10), columns=['loss'])
results = pd.DataFrame()

# Iterate through test sets
for i in range(len(strata_idxs)):

    test_idx = strata_idxs[i]

    if i + 1 < len(strata_idxs):
        valid_idx = strata_idxs[i + 1]
        train_idx = np.concatenate(np.delete(strata_idxs, [i, i + 1]))
    else:
        valid_idx = strata_idxs[0]
        train_idx = np.concatenate(np.delete(strata_idxs, [i, 0]))

    print(f'\nStarting trial {i + 1}\n')

    # Obtain weights by ols on training and validation data
    X = mat_from_idx(np.concatenate((train_idx, valid_idx)))
    y = np.array(df['age'].iloc[np.concatenate((train_idx, valid_idx))]).reshape((-1, 1))
    w = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    # Compute predicted values for test data
    y_pred = (np.array(mat_from_idx(test_idx)) @ w).flatten()

    # Store test loss in dataframe
    summary['loss'].iloc[i] = tf.keras.losses.mean_squared_error(
        y_pred, np.array(df['age'].iloc[test_idx]).astype(float)
    ).numpy()

    # Store individual results in dataframe
    result = pd.DataFrame(
        {'filename': df['filename'].iloc[test_idx],
         'age': df['age'].iloc[test_idx],
         'length': df['length'].iloc[test_idx],
         'sex': df['sex'].iloc[test_idx],
         'y': np.array(y_pred).flatten()}
    )

    if i == 0:
        results = result
    else:
        results = pd.merge(results, result, how='outer')

# Save dataframes to file
summary.to_csv(r'C:\Users\iverm\Desktop\Results\summary.csv')
results.to_csv(r'C:\Users\iverm\Desktop\Results\results.csv')
