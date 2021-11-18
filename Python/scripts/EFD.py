# -*- coding: utf-8 -*-
"""
Example of obtaining elliptical fourier descriptors (EFD's)
of an otolith image.

Created on Thu Jul 22 11:53:00 2021

@author: iverm
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from modules.efd import contour_img, chain_code

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
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(blurred, 'gray')
ax1.set_title('Blurred image')
ax1.axis('off')
ax2.imshow(thresholded, 'gray')
ax2.set_title('Thresholded image')
ax2.axis('off')

# extract the boundary of the otolith
points, boundary = contour_img(thresholded, 0)

# display boundary image
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(thresholded, 'gray')
ax1.set_title('Thresholded image')
ax1.axis('off')
ax2.imshow(boundary, 'gray')
ax2.set_title('Object boundary')
ax2.axis('off')

# construct a freeman chain of the contour pixels
code = chain_code(points)

# Time used for each step in the Freeman chain.
# Eqs. given by Giardina & Kuhl (1982).
delta_t = 1 + ((np.sqrt(2) - 1) / 2)*(1 - (-1)**code)

t = np.cumsum(delta_t)                 # accumulated time vector
t_min = np.concatenate(([0], t[:-1]))  # shifted time vector
T = t[-1]                              # fundamental period

# Changes in projections along the x and y axis with
# eqs. given by Giardina & Kuhl (1982).
# Here the x and y axis are oriented by the standard cartesian system.
delta_x = np.sign(6 - code) * np.sign(2 - code)
delta_y = np.sign(4 - code) * np.sign(code)


# Functions for computing the Fourier coefficents for x and y
# with eqs. given by Giardina & Kuhl (1982).
def a(n):
    return T * np.sum(delta_x * (np.cos(2*n*np.pi*t/T) - np.cos(2*n*np.pi*t_min/T)) / delta_t) / (2*n**2*np.pi**2)


def b(n):
    return T * np.sum(delta_x * (np.sin(2*n*np.pi*t/T) - np.sin(2*n*np.pi*t_min/T)) / delta_t) / (2*n**2*np.pi**2)


def c(n):
    return T * np.sum(delta_y * (np.cos(2*n*np.pi*t/T) - np.cos(2*n*np.pi*t_min/T)) / delta_t) / (2*n**2*np.pi**2)


def d(n):
    return T * np.sum(delta_y * (np.sin(2*n*np.pi*t/T) - np.sin(2*n*np.pi*t_min/T)) / delta_t) / (2*n**2*np.pi**2)


# Computing the DC components for x and y using eqs. from Giardina & Kuhl.            
epsilon = np.cumsum(np.concatenate(([0], delta_x[:-1]))) - delta_x * np.cumsum(t_min) / delta_t
delta = np.cumsum(np.concatenate(([0], delta_y[:-1]))) - delta_y * np.cumsum(t_min) / delta_t

A0 = np.sum(delta_x*(t**2-t_min**2)/(2*delta_t)+epsilon*(t-t_min))/T
C0 = np.sum(delta_y*(t**2-t_min**2)/(2*delta_t)+delta*(t-t_min))/T

# Functions returning x(t) and y(t) for the nth harmonic.
x_terms = lambda n: a(n)*np.cos(2*n*np.pi*t/T)+b(n)*np.sin(2*n*np.pi*t/T)
y_terms = lambda n: c(n)*np.cos(2*n*np.pi*t/T)+d(n)*np.sin(2*n*np.pi*t/T)


# Functions returning the total sum of x(t) and y(t) given n harmonics.
def x(n):
    output = A0
    for i in range(1, n + 1):
        output += x_terms(i)
    return output


def y(n):
    output = C0
    for i in range(1, n + 1):
        output += y_terms(i)
    return output


# Choices of n to display
n = [1, 2, 5, 10, 20, 100]

# plot contour approximations
_, axes = plt.subplots(3, 3)

for i, ax in enumerate(axes.flatten()):
    if i == 0:
        ax.imshow(image, 'gray')
        ax.set_title('Original image')
        ax.axis('off')
    elif i == 1:
        ax.imshow(thresholded, 'gray')
        ax.set_title('Binary image')
        ax.axis('off')
    elif i == 2:
        ax.imshow(boundary, 'gray')
        ax.set_title('Contour image')
        ax.axis('off')
    else:
        ax.plot(x(n[i-3]), y(n[i-3]))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{n[i-3]} Fourier coefficients')
        ax.axis('off')
