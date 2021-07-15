# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:46:54 2021

@author: iverm
"""
from imageloader import imageloader

path = r'C:\Users\iverm\Google Drive\Masteroppgave\Data\Torskeotolitter\standard'

images = imageloader(path, (128, 128), (0.6, 0.2, 0.2), seed=None)

for i in range(len(images)):
    for j in range(i + 1, len(images)):
        for image1 in images[i]['images']:
            for image2 in images[j]['images']:
                assert (image1 == image2).all() == False, 'error'
                
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        for filename1 in images[i]['filenames']:
            for filename2 in images[j]['filenames']:
                assert (filename1 == filename2) == False, 'error'