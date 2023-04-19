import pandas as pd

df = pd.read_csv("data/dataframe.csv")

path_to_images = "data/images256.npy"

img_known_sex = np.load(path_to_images)[np.where(df["sex"] != "unknown")]
img_unknown_sex = np.load(path_to_images)[np.where(df["sex"] == "unknown")]

# Create stratified indices for selecting datasets for training etc.
strata_idxs = stratified_idxs(
    df["age"].iloc[np.where(df["sex"] != "unknown")], 10, seed=1234
)
