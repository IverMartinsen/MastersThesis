# -*- coding: utf-8 -*-
"""
Function for displaying confusion matrices.

Created on Fri Jul 23 14:45:14 2021

@author: iverm
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(
        labels, predictions, class_names=None, figsize=(12, 8), 
        cmap=plt.cm.RdYlGn):
    '''
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

    '''
    
    # computes an m-by-m array
    conf_mat = tf.math.confusion_matrix(
        labels.flatten(), predictions.flatten()).numpy()
    
    # m-by-m array to hold accuracies
    conf_norm = np.zeros_like(conf_mat, dtype=float)

    # compute accuracy for each cell
    for i, row in enumerate(conf_mat):  # for each row
        totals = np.sum(row)            # total #samples for class i
        for j, count in enumerate(row): # for each cell in row
            if i == j:
                conf_norm[i, j] = count / totals # fraction of true pos.
                # rescaled to (0.5, 1)
                conf_norm[i, j] = conf_norm[i, j] / 2 + 0.5 
            else:
                conf_norm[i, j] = 1 - count / totals # fraction of true neg.
                # rescaled to (0, 0.5)
                conf_norm[i, j] = conf_norm[i, j] / 2 
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    mappable = ax.imshow(
        conf_norm,
        cmap=cmap,
        interpolation='nearest',
        vmin = 0,
        vmax = 1)

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