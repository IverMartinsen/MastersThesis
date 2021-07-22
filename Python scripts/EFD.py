# -*- coding: utf-8 -*-
"""
Example of obtaining elliptical fourier discriptors (EFD's) 
of an otolith image.

Created on Thu Jul 22 11:53:00 2021

@author: iverm
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# import and normalize image
path = (r'C:\Users\iverm\Google Drive\Masteroppgave' + 
        r'\Data\Torskeotolitter\raw\cc\fil0010.jpg')

image = np.array(Image.open(path)) / 255.

# blur image using a gaussian kernel
blurred = cv.GaussianBlur(image, (7, 7), 1)

# display effect of blurring
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image, 'gray')
ax1.set_title('Original image')
ax1.axis('off')
ax2.imshow(blurred, 'gray')
ax2.set_title('Blurred image')
ax2.axis('off')

# threshold image
_, thresholded = cv.threshold(blurred, 0.5, 1, cv.THRESH_BINARY)

# display effect of thresholding
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(blurred, 'gray')
ax1.set_title('Blurred image')
ax1.axis('off')
ax2.imshow(thresholded, 'gray')
ax2.set_title('Thresholded image')
ax2.axis('off')
