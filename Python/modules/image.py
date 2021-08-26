# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:48:31 2021

@author: iverm
"""
import numpy as np

def normalize(img, lower=0, upper=1):
    '''
    Normalize input image to the given range.
    Assumes equal scaling for all images in input tensor.
    Assumes RGB or grayscale format on input.
    '''
    return (img-np.min(img))*(upper-lower)/(np.max(img)-np.min(img))+lower