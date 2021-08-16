# -*- coding: utf-8 -*-
"""
Tools for image processing. 

Created on Thu Jul 22 15:19:23 2021

@author: iverm
"""
import numpy as np

def contour_img(img, target):
    '''
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

    '''    
    
    # starting at the uppermost-leftmost pixel with target value
    b0 = np.array(np.where(img == target)).transpose()[0,:]
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
        start = np.where(np.all(neighbours == (c - b), axis = 1))[0][0]
    
        x_neighbours_current = x_neighbours[start:] + x_neighbours[:start]
        y_neighbours_current = y_neighbours[start:] + y_neighbours[:start]
    
        # find first neighbouring pixel with target value
        first = np.where(img[b[0] + x_neighbours_current , 
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
    '''
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

    '''
    
    diff = np.vstack((points[1:], points[:1])) - points

    V = np.zeros(diff.shape[0])
    
    V[np.where(np.all(diff == [+0, +1], axis = 1))] = 0
    V[np.where(np.all(diff == [-1, +1], axis = 1))] = 1
    V[np.where(np.all(diff == [-1, +0], axis = 1))] = 2
    V[np.where(np.all(diff == [-1, -1], axis = 1))] = 3
    V[np.where(np.all(diff == [+0, -1], axis = 1))] = 4
    V[np.where(np.all(diff == [+1, -1], axis = 1))] = 5
    V[np.where(np.all(diff == [+1, +0], axis = 1))] = 6
    V[np.where(np.all(diff == [+1, +1], axis = 1))] = 7
    
    return V



def normalize(img, lower=0, upper=1):
    '''
    Normalize input image to the given range.
    Assumes equal scaling for all images in input tensor.
    Assumes RGB or grayscale format on input.
    '''
    return (img-np.min(img))*(upper-lower)/(np.max(img)-np.min(img))+lower
    
    
    