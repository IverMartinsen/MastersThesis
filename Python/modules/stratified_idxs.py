import numpy as np


def stratified_idxs(labels, splits):
    """
    Produce stratified subsets of indices from labels.

    Parameters
    ----------
    labels : list-like sequence of class labels(int)
    splits : number of splits(int) or fractions(list/tuple/array)

    Returns
    -------
    tuple of numpy.arrays with indices for each subset
    """
    # create list of splits
    if type(splits) == int:
        num_subsets = splits
        splits = np.repeat(1 / splits, num_subsets)
    else:
        try:
            num_subsets = len(splits)
        except TypeError:
            print('splits must be int or list-like')

    class_labels = np.unique(labels)

    subsets = {}

    # for each class, select same proportion of images for all subsets
    for i, class_label in enumerate(class_labels):

        class_idxs = np.where(labels == class_label)[0]
        class_size = len(class_idxs)

        subset_idx = np.round(
            np.cumsum(class_size * np.array(splits))).astype('int')

        for j in range(num_subsets):

            if i == 0:
                if j == 0:
                    subsets[j] = class_idxs[:subset_idx[j]]
                else:
                    subsets[j] = class_idxs[subset_idx[j - 1]:subset_idx[j]]

            else:
                if j == 0:
                    subsets[j] = np.concatenate(
                        (subsets[j], class_idxs[:subset_idx[j]])
                    )

                else:
                    subsets[j] = np.concatenate(
                        (subsets[j],
                         class_idxs[subset_idx[j - 1]:subset_idx[j]])
                    )

    return tuple(subsets.values())
