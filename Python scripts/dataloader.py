'''
Contains:
    get_label()    : support function for dataloader 
    process_path() : support function for dataloader
    dataloader()   : function for loading images from path
'''

import pathlib
import numpy as np
import tensorflow as tf
import os

def get_label(file_path, class_names):
    '''
    Takes an image file path and returns class label.

    Parameters
    ----------
    file_path : str
        Path to file.
    class_names : numpy.ndarray
        Array of class names.

    Returns
    -------
    int
        Class label.

    '''
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)

def process_path(file_path, image_size, channels, keep_aspect, class_names):
    '''
    Takes an image file path and returns a tuple of (tensor, label)
    with each image in tensor resized to image_size.

    Parameters
    ----------
    file_path : str
        Path to file.
    image_size : tuple
        (height, width).
    channels : int
        Number of channels.
    keep_aspect : bool
        Status of keeping aspect when resizing.
    class_names : numpy.ndarray
        Array of class names.

    Returns
    -------
    img : Tensor
        Image.
    label : int
        Class label.

    '''
    # get image label
    label = get_label(file_path, class_names)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=channels)
    # resize image to image_size
    if keep_aspect:
        img = tf.image.resize_with_pad(img, image_size[0], image_size[1])
    else:    
        img = tf.image.resize(img, image_size)
    
    filename = tf.strings.split(file_path, os.path.sep)[-1]
    img.filename = filename
    return img, label, filename

def dataloader(file_path, image_size, channels, splits=1, keep_aspect=False,
               batch_size=32):
    '''
    Loads images from path to class folders.

    Parameters
    ----------
    file_path : str
        Path to class folders.
        Must contain one subfolder with images for each class. 
    image_size : tuple
        Desired output size of images.
    channels : int
        Number of channels in input images.
    splits : int or list-like, optional
        If int, splits set into subsets of equal sizes.
        If list, splits set into subsets of sizes given by fractions in list.
        The default is 1.
    keep_aspect : bool, optional
        If aspect ratio should be kept during resizing.
        The default is False.
    batch_size : int, optional
        Batch size. The default is 32.

    Returns
    -------
    tf.data.Dataset
        Dataset of images.

    '''
    data_dir = pathlib.Path(file_path)

    class_names = np.array([item.name for item in data_dir.glob('*')])

    num_images = len(list(data_dir.glob('*/*.jpg')))
    num_classes = len(class_names)
    
    print(f'Total number of images: {num_images}')
    print(f'Total number of classes: {num_classes}')
    
    label_names = {}
    files_by_class = {}
    
    for i, class_name in enumerate(class_names):
        # map labels with class names
        label_names[i] = class_name
        # store filenames as tf.Dataset
        files_by_class[class_name] = (tf.data.Dataset.list_files(
            str(data_dir/class_name/'*'), shuffle=False))

    if type(splits) == int:
        num_subsets = splits
        splits = np.repeat(1 / splits, num_subsets)
    else:
        try:
            num_subsets = len(splits)
        except TypeError:
            print('splits must be int or list-like')
        
    subsets = {}
    
    # for each class, select same proportion for all subsets
    for i, class_name in enumerate(class_names):
        
        class_size = len(files_by_class[class_name])
        list_class = files_by_class[class_name]

        for j in range(num_subsets):
            
            subset_size = int(class_size*splits[j])
            
            if i == 0:
                if j < num_subsets - 1:
                    subsets[j] = list_class.skip(subset_size*j).take(subset_size)
                else:
                    subsets[j] = list_class.skip(subset_size*j)
            else:
                if j < num_subsets - 1:
                    subsets[j] = (subsets[j].concatenate(
                        list_class.skip(subset_size*j).take(subset_size)))
                else:
                    subsets[j] = (subsets[j].concatenate(
                        list_class.skip(subset_size*j)))
                    
    # shuffle filenames for each subset
    for j in range(num_subsets):
        
        subsets[j] = subsets[j].shuffle(
            len(subsets[j]), reshuffle_each_iteration=False)
        
        subsets[j] = subsets[j].map(lambda x: process_path(
            x, image_size, channels, keep_aspect, class_names),
            num_parallel_calls=tf.data.AUTOTUNE)

        # configure for performance
        subsets[j] = subsets[j].cache()
        subsets[j] = subsets[j].batch(batch_size)
        subsets[j] = subsets[j].prefetch(buffer_size=tf.data.AUTOTUNE)
        
        filenames = [[item.decode() for item in element.numpy()]
                     for element in list(subsets[j].map(lambda x, y, z: z))]
        
        subsets[j] = subsets[j].map(lambda x, y, z: (x, y))

        subsets[j].filenames = filenames
        subsets[j].class_names = label_names
    
    assertlist = np.array([])
    for sets in tuple(subsets.values()):
        names = np.concatenate([i for i in sets.filenames])
        assertlist = np.concatenate((assertlist, names))
    
    print(np.unique(assertlist).shape)
    
    

    return tuple(subsets.values())