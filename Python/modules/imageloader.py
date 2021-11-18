'''
Module for loading images from folder.
'''

import pathlib
import numpy as np
import os
from PIL import Image
from modules.image import normalize


class ImageGenerator:

    def __init__(
            self, file_paths, image_size, class_names, mode, normalize=False
    ):
        self.file_paths = file_paths
        self.image_size = image_size
        self.class_names = class_names
        self.mode = mode
        self.normalize = normalize

    def __getitem__(self, key):
        labels = np.array([get_label(file_path, self.class_names) for
                           file_path in self.file_paths])

        images = np.stack(
            [np.array(
                Image.open(
                    file_path).convert(
                    self.mode).resize(
                    self.image_size)).astype(
                float) for file_path in self.file_paths])

        if self.normalize:
            images = normalize(images)

        filenames = [
            file_path.split(os.path.sep)[-1] for file_path in self.file_paths
        ]

        data = {'images': images, 'labels': labels, 'filenames': filenames}

        return data[key]


def get_label(file_path, class_names):
    '''
    Takes an image file path and returns class label.

    Parameters
    ----------
    file_path : str
        Path to file.
    class_names : numpy.ndarray
        Array of class names.

    Returns
    -------
    int
        Class label.

    '''
    # convert the path to a list of path components
    parts = file_path.split(os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    assert (np.sum(one_hot) == 1)
    # Integer encode the label
    return np.argmax(one_hot)


def load_images(file_path, image_size, splits=1, seed=None, mode='L'):
    '''
    Loads images from path to class folders.

    Parameters
    ----------
    file_path : str
        Path to class folders.
        Must contain one subfolder with images for each class.
    image_size : tuple
        Desired output size of images.
    splits : int or list-like, optional
        If int, splits set into subsets of equal sizes.
        If list, splits set into subsets of sizes given by fractions in list.
        The default is 1.
    seed : int, optional
        Seed used for shuffling.
        The default is None.
    mode : str
        The requested mode.
        Use L for grayscale, RGB for color. The default is L.

    Returns
    -------
    tuple
        Tuple of ImageGenerator objects.

    '''

    assert os.path.isdir(file_path), f"{file_path} does not exist"

    filetype = 'jpg'

    rng = np.random.RandomState(seed)

    data_dir = pathlib.Path(file_path)

    class_names = np.array([item.name for item in data_dir.glob('*')])

    num_images = len(list(data_dir.glob('*/*.' + filetype)))
    num_classes = len(class_names)

    print(f'Total number of images: {num_images}')
    print(f'Total number of classes: {num_classes}')

    label_names = {}
    files_by_class = {}

    # list and shuffle filenames for each class
    for i, class_name in enumerate(class_names):
        # map labels with class names
        label_names[i] = class_name
        # store filepaths in list
        files_by_class[class_name] = [
            str(data_dir / class_name / filename) for
            filename in os.listdir(str(data_dir / class_name))]

        rng.shuffle(files_by_class[class_name])

    # create list of splits
    if type(splits) == int:
        num_subsets = splits
        splits = np.repeat(1 / splits, num_subsets)
    else:
        try:
            num_subsets = len(splits)
        except TypeError:
            print('splits must be int or list-like')

    subsets = {}

    # for each class, select same proportion of images for all subsets
    for i, class_name in enumerate(class_names):

        class_size = len(files_by_class[class_name])
        list_class = files_by_class[class_name]

        subset_idx = np.round(
            np.cumsum(class_size * np.array(splits))).astype('int')

        for j in range(num_subsets):

            if i == 0:
                if j == 0:
                    subsets[j] = list_class[:subset_idx[j]]
                else:
                    subsets[j] = list_class[subset_idx[j - 1]:subset_idx[j]]

            else:
                if j == 0:
                    subsets[j] = np.concatenate(
                        (subsets[j], list_class[:subset_idx[j]]))

                else:
                    subsets[j] = np.concatenate(
                        (subsets[j],
                         list_class[subset_idx[j - 1]:subset_idx[j]]))

    # print number of images in splits
    print('----------------------------')
    for size in np.unique([len(subset) for subset in subsets.values()]):
        num = np.sum(
            np.array([len(subset) for subset in subsets.values()]) == size)
        if num == 1:
            print(f'{num} subset with {size} images')
        else:
            print(f'{num} subsets with {size} images')

    # shuffle filenames for each subset
    for j in range(num_subsets):
        rng.shuffle(subsets[j])

        subsets[j] = ImageGenerator(subsets[j], image_size, class_names, mode)

    return tuple(subsets.values())