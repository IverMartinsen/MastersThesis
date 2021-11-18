# -*- coding: utf-8 -*-
"""
Collection of utility functions

    normalize()
    confusion_matrix()
    read_file()
    move_files()
    list_files()
    images_to_array()
    add_jpeg()
    contour_img()
    chain_code()
    stratified_idxs()

Created on Fri Jul 23 14:45:14 2021

@author: iverm
"""

import os
import pandas as pd
import shutil
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(img, lower=0, upper=1):
    """
    Normalize input image to the given range.
    Assumes equal scaling for all images in input tensor.
    Assumes RGB or grayscale format on input.
    """
    return (img-np.min(img))*(upper-lower)/(np.max(img)-np.min(img))+lower


def confusion_matrix(labels, predictions, class_names=None, figsize=(12, 8), cmap=plt.cm.RdYlGn):
    """
    Displays confusion matrix using provided labels and predictions.

    Parameters
    ----------
    labels : list or numpy arrays
        True labels.
    predictions : list or numpy arrays
        Predicted labels.
    class_names : list of strings, optional
        List of class names. The default is None.
    figsize : tuple, optional
        Size of figure. The default is (12, 8).
    cmap : matplotlib.pyplot.cm, optional
        Colormap used in background. The default is plt.cm.RdYlGn.

    Returns
    -------
    None.

    """
    
    # computes an m-by-m array
    conf_mat = tf.math.confusion_matrix(
        labels.flatten(), predictions.flatten()).numpy()
    
    # m-by-m array to hold accuracies
    conf_norm = np.zeros_like(conf_mat, dtype=float)

    # compute accuracy for each cell
    for i, row in enumerate(conf_mat):   # for each row
        totals = np.sum(row)             # total #samples for class i
        for j, count in enumerate(row):  # for each cell in row
            if i == j:
                conf_norm[i, j] = count / totals  # fraction of true pos.
                # rescaled to (0.5, 1)
                conf_norm[i, j] = conf_norm[i, j] / 2 + 0.5 
            else:
                conf_norm[i, j] = 1 - count / totals  # fraction of true neg.
                # rescaled to (0, 0.5)
                conf_norm[i, j] = conf_norm[i, j] / 2 
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    mappable = ax.imshow(
        conf_norm,
        cmap=cmap,
        interpolation='nearest',
        vmin=0,
        vmax=1)

    height, width = conf_mat.shape

    # annotate every number in conf_mat
    for x in range(height):
        for y in range(width):
            ax.annotate(
                str(conf_mat[x][y]), xy=(y, x), 
                horizontalalignment='center',
                verticalalignment='center',
                size=16,
                weight='bold')

    fig.colorbar(mappable)
    
    # set class names if not given
    if class_names is None:
        class_names = ['class ' + str(i + 1) for i in range(height)]
      
    # set layout for ticks
    ax.xaxis.tick_top()
    ax.set_xticks(range(width))
    ax.set_xticklabels(class_names, fontsize=18)
    plt.yticks(range(height), class_names, size=18, rotation=90, va='center')
    
    # set layout for labels
    ax.set_xlabel('Predictions', size=28, labelpad=15)
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('True labels', size=28, labelpad=15)

    plt.show()


def read_file(file):
    """
    Reads numbers from txt-file.

    Parameters
    ----------
    file : str
        TXT FILE OF NUMBERS.

    Returns
    -------
    numpy.ndarray
        ARRAY OF FILE CONTENT.

    """
    return np.array(pd.read_csv(file, header=None)).reshape(-1)


def move_files(filenames, source, destination):
    """
    Move files from source to destination.

    Parameters
    ----------
    filenames : list
        LIST OF FILE NAMES.
    source : str
        PATH TO SOURCE.
    destination : str
        PATH TO DESTINATION.

    Returns
    -------
    None.

    """
    for file in filenames:
        try:
            shutil.move(os.path.join(source, file), os.path.join(destination, file))
        except FileNotFoundError:
            print(file + ' not found')


def list_files(folder, idxs):
    """
    List filenames in folder that matches numbers in idxs.

    Parameters
    ----------
    folder : str
        PATH TO FOLDER.
    idxs : numpy.ndarray
        ARRAY OF INDICES.

    Returns
    -------
    filenames : list
        LIST OF FILENAMES.

    """
    filenames = []

    for filename in os.listdir(folder):
        for idx in idxs:
            if idx == int(re.findall(r'\d+', filename)[0].lstrip('0')):
                filenames.append(filename)

    return filenames


def images_to_array(folder, height, width, channels):
    """
    Convert images in folder into tensor.

    Parameters
    ----------
    folder : str
        PATH TO IMAGES.
    height : int
        DESIRED IMAGE HEIGHT.
    width : int
        DESIRED IMAGE WIDTH.
    channels : int
        NUMBER OF CHANNELS IN IMAGES

    Returns
    -------
    images : numpy.ndarray
        TENSOR OF IMAGES (N x H x W x C).

    """
    filenames = os.listdir(folder)
    num_images = len(filenames)
    images = np.zeros((num_images, height, width))

    for i in range(len(filenames)):
        image = np.array(Image.open(folder + filenames[i]), dtype=np.float32)
        try:
            images[i, ...] = image
        except ValueError:
            image = image.resize((height, width))

    return images.reshape(num_images, height, width, channels)


def add_jpeg(folder):
    """
    If lacking, adds .jpeg extension to all images in folder.

    Parameters
    ----------
    folder : str
        PATH TO FOLDER.

    Returns
    -------
    None.

    """
    for filename in os.listdir(folder):
        if filename[-3:] != 'jpg':
            os.rename(folder + filename, folder + filename + '.jpg')


def contour_img(img, target):
    """
    Returns boundary image from a binary input image.
    Based on the Moore Boundary Tracing Algorithm.

    Parameters
    ----------
    img : numpy.ndarray
        Binary input image.
    target : int
        Intensity value of object, must be 0 or 1.

    Returns
    -------
    boundary_points, boundary_img : tuple
        Boundary points and boundary image.

    """

    # starting at the uppermost-leftmost pixel with target value
    b0 = np.array(np.where(img == target)).transpose()[0, :]
    c0 = b0 - [0, 1]

    # matrix of relative 8-neighbours positions starting west
    x_neighbours = [0, -1, -1, -1, 0, 1, 1, 1]
    y_neighbours = [-1, -1, 0, 1, 1, 1, 0, -1]
    neighbours = np.vstack((x_neighbours, y_neighbours)).transpose()

    condition = True

    b = np.copy(b0)
    c = np.copy(c0)

    while condition:

        # add b to the list of boundary points
        try:
            boundary_points = np.vstack((boundary_points, b))
        except NameError:
            boundary_points = b0

        # starting position for iterating over neighbours
        start = np.where(np.all(neighbours == (c - b), axis=1))[0][0]

        x_neighbours_current = x_neighbours[start:] + x_neighbours[:start]
        y_neighbours_current = y_neighbours[start:] + y_neighbours[:start]

        # find first neighbouring pixel with target value
        first = np.where(img[b[0] + x_neighbours_current,
                             b[1] + y_neighbours_current] == target)[0][0]

        # update c
        c = b + [x_neighbours_current[first - 1],
                 y_neighbours_current[first - 1]]

        # update b
        b += [x_neighbours_current[first], y_neighbours_current[first]]

        # check for stopping criterion
        condition = np.all(b == b0) == False

    # create boundary image
    boundary_img = np.zeros_like(img)
    boundary_img[boundary_points[:, 0], boundary_points[:, 1]] = 1

    return boundary_points, boundary_img


def chain_code(points):
    """
    Returns the Freeman chain code from an n x 2 array of
    boundary points.

    Parameters
    ----------
    points : numpy.ndarray
        Boundary points.

    Returns
    -------
    V : numpy.ndarray
        Chain code sequence.

    """

    diff = np.vstack((points[1:], points[:1])) - points

    v = np.zeros(diff.shape[0])

    v[np.where(np.all(diff == [+0, +1], axis=1))] = 0
    v[np.where(np.all(diff == [-1, +1], axis=1))] = 1
    v[np.where(np.all(diff == [-1, +0], axis=1))] = 2
    v[np.where(np.all(diff == [-1, -1], axis=1))] = 3
    v[np.where(np.all(diff == [+0, -1], axis=1))] = 4
    v[np.where(np.all(diff == [+1, -1], axis=1))] = 5
    v[np.where(np.all(diff == [+1, +0], axis=1))] = 6
    v[np.where(np.all(diff == [+1, +1], axis=1))] = 7

    return v


def stratified_idxs(labels, splits, seed = None):
    """
    Produce stratified subsets of indices from labels.

    Parameters
    ----------
    labels : list-like sequence of class labels(int)
    splits : number of splits(int) or fractions(list/tuple/array)
    seed : Optional, int. Seed for shuffling. The default is None.

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

    rng = np.random.default_rng(seed=seed)

    # for each class, select same proportion of images for all subsets
    for i, class_label in enumerate(class_labels):

        class_idxs = np.where(labels == class_label)[0]
        class_size = len(class_idxs)

        rng.shuffle(class_idxs)

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

            rng.shuffle(subsets[j])

    return tuple(subsets.values())


