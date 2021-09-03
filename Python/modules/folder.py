'''Functions for sorting and organizing files.

Functions:
    read_file()
    move_files()
    list_files()
    images_to_array()
    add_jpeg()
'''
import os
import numpy as np
import pandas as pd
import shutil
import re
from PIL import Image

def read_file(file):
    '''
    Reads numbers from txt-file.

    Parameters
    ----------
    file : str
        TXT FILE OF NUMBERS.

    Returns
    -------
    numpy.ndarray
        ARRAY OF FILE CONTENT.

    '''
    return np.array(pd.read_csv(file, header = None)).reshape(-1)


def move_files(filenames, source, destination):
    '''
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

    '''
    for file in filenames:
        try:
            shutil.move(os.path.join(source, file), os.path.join(destination, file))
        except FileNotFoundError:
            print(file + ' not found')


def list_files(folder, idxs):
    '''
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

    '''
    filenames = []
    
    for filename in os.listdir(folder):
        for idx in idxs:
            if idx == int(re.findall(r'\d+', filename)[0].lstrip('0')):
                filenames.append(filename)
    
    return filenames

def images_to_array(folder, height, width, channels):
    '''
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

    '''
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
    '''
    If lacking, adds .jpeg extension to all images in folder.

    Parameters
    ----------
    folder : str
        PATH TO FOLDER.

    Returns
    -------
    None.

    '''        
    for filename in os.listdir(folder):
        if filename[-3:] != 'jpg':
            os.rename(folder + filename, folder + filename + '.jpg')
        

