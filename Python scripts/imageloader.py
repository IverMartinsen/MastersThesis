'''
Contains:
    get_label()    : support function for dataloader 
    process_path() : support function for dataloader
    dataloader()   : function for loading images from path
'''

import pathlib
import numpy as np
import os
from PIL import Image

class ImageGenerator:
    def __init__(self, file_paths, image_size, class_names):
        self.file_paths = file_paths
        self.image_size = image_size
        self.class_names = class_names
    
    def __getitem__(self, key):
        labels = np.array([get_label(file_path, self.class_names) for 
                  file_path in self.file_paths])
        images = np.moveaxis(np.dstack([np.array(Image.open(file_path).resize(self.image_size)).astype(float) for file_path in self.file_paths]), -1, 0)
        filenames = [file_path.split(os.path.sep)[-1] for file_path in self.file_paths]
        
        test = {'images':images, 'labels':labels, 'filenames':filenames}
        return test[key]


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
    parts = file_path.split(os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    assert(np.sum(one_hot) == 1)
    # Integer encode the label
    return np.argmax(one_hot)

def process_path(file_path, image_size, class_names):
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
    label = get_label(file_path, class_names)
    img = np.array(Image.open(file_path).resize(image_size)).astype(float)
    filename = file_path.split(os.path.sep)[-1]
    
    return img, label, filename

def imageloader(file_path, image_size, splits=1, seed=None):
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
    
    filetype='jpg'
    
    rng = np.random.RandomState(seed)
    
    data_dir = pathlib.Path(file_path)

    class_names = np.array([item.name for item in data_dir.glob('*')])
    
    num_images = len(list(data_dir.glob('*/*.' + filetype)))
    num_classes = len(class_names)
    
    print(f'Total number of images: {num_images}')
    print(f'Total number of classes: {num_classes}')
    
    label_names = {}
    files_by_class = {}
    
    # list and shuffle filenames for each class
    for i, class_name in enumerate(class_names):
        # map labels with class names
        label_names[i] = class_name
        # store filepaths in list
        files_by_class[class_name] = [
            str(data_dir/class_name/filename) for 
            filename in os.listdir(str(data_dir/class_name))]
        
        rng.shuffle(files_by_class[class_name])
 
    # create list of splits
    if type(splits) == int:
        num_subsets = splits
        splits = np.repeat(1 / splits, num_subsets)
    else:
        try:
            num_subsets = len(splits)
        except TypeError:
            print('splits must be int or list-like')
        
    subsets = {}
    
    # for each class, select same proportion of images for all subsets
    for i, class_name in enumerate(class_names):
        
        class_size = len(files_by_class[class_name])
        list_class = files_by_class[class_name]

        for j in range(num_subsets):
            
            subset_size = class_size*splits[j]
            
            if i == 0:
                subsets[j] = list_class[
                    int(np.round(subset_size*j)):
                        int(np.round(subset_size*(j + 1)))]
            else:
                subsets[j] = np.concatenate((subsets[j],
                    list_class[
                        int(np.round(subset_size*j)):
                            int(np.round(subset_size*(j + 1)))]))
    
    # print number of images in splits
    print('----------------------------')
    for size in np.unique([len(subset) for subset in subsets.values()]):
        num = np.sum(
            np.array([len(subset) for subset in subsets.values()]) == size)
        if num == 1:
            print(f'{num} subset with {size} images')
        else:
            print(f'{num} subsets with {size} images')
    
    
    # shuffle filenames for each subset
    for j in range(num_subsets):
        
        rng.shuffle(subsets[j])        
        
        subsets[j] = ImageGenerator(subsets[j], image_size, class_names)  
            
    return tuple(subsets.values())