# -*- coding: utf-8 -*-
"""
Test function for load_images.

Created on Sun Jun 13 13:46:54 2021

@author: iverm
"""
import os
from imageloader import imageloader

def test_imageloader():
    '''
    Test function for imageloader. Tests for independence between sets.

    Returns
    -------
    None.

    '''    
    THIS_DIR = os.path.dirname(os.path.abspath('test_imageloader'))

    path = os.path.join(THIS_DIR, 'testdata')

    images = imageloader(path, (128, 128), (0.7, 0.2, 0.1), seed=None)

    # assert that no image appears in more than one set
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            for image_in_i in images[i]['images']:
                for image_in_j in images[j]['images']:
                    assert (image_in_i == image_in_j).all() == False, (
                        f'Warning: an image in set {i} also detected in set {j}!')
    
    # assert that no filename appears in more than one set
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            for filename1 in images[i]['filenames']:
                for filename2 in images[j]['filenames']:
                    assert (filename1 == filename2) == False, (
                        f'Warning: {filename1} detected in set {i} and set {j}!')

if __name__ == '__main__':
    test_imageloader()
    print('Test passed!')