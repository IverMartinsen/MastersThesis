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
from image_tools import contour_img

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

# extract the boundary of the otolith
points, boundary = contour_img(thresholded, 0)

# display boundary image
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(thresholded, 'gray')
ax1.set_title('Thresholded image')
ax1.axis('off')
ax2.imshow(boundary, 'gray')
ax2.set_title('Object boundary')
ax2.axis('off')

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
