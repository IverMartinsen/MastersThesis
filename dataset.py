import pathlib
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self, image_size, channels=1, keep_aspect=False):
        # if has the dataset been split
        self._is_split = False
        self.channels = channels
        self.keep_aspect = False
        self.image_size = image_size

    def load(self, file_path):
        '''
        Load filenames from file_path, count images and classes and
        store label/names and names/datasets in dicts.

        Parameters
        ----------
        file_path : str
            Path to image folders.

        Returns
        -------
        None.

        '''            
        self._file_path = file_path
        self._data_dir = pathlib.Path(file_path)

        self.image_count = len(list(self._data_dir.glob('*/*.jpg')))
        self.class_names = (np.array(
            [item.name for item in self._data_dir.glob('*')]))
        self.class_count = len(self.class_names)
        self._label_to_name = {}
        self._class_to_list = {}
        
        for i, class_name in enumerate(self.class_names):
      
            self._label_to_name[i] = class_name
            # store filenames as tf.Dataset
            self._class_to_list[class_name] = (tf.data.Dataset.list_files(
                str(self._data_dir/class_name/'*'), shuffle=False))
            # shuffle filenames
            self._class_to_list[class_name] = (
                self._class_to_list[class_name].shuffle(
                    len(self._class_to_list[class_name]),
                    reshuffle_each_iteration=False))

    def split(self, split):
        '''
        Split data into training, validation and test set.

        Parameters
        ----------
        split : tuple
            (tr_fraction, va_fraction, te_fraction).

        Returns
        -------
        None.

        '''
        tr_spl, va_spl, te_spl = split

        # for each class, select same proportion
        # for train_ds, valid_ds and test_ds        
        for i, class_name in enumerate(self.class_names):
            
            class_size = len(self._class_to_list[class_name])
            list_class = self._class_to_list[class_name]
            
            valid_size = int(class_size*va_spl)
            test_size = int(class_size*te_spl)

            if i == 0:
                self.valid_ds = list_class.take(valid_size)
                self.test_ds = list_class.skip(valid_size).take(test_size)
                self.train_ds = list_class.skip(valid_size + test_size)
            else:
                self.valid_ds = (
                    self.valid_ds.concatenate(
                        list_class.take(valid_size)))
                self.test_ds = (
                    self.test_ds.concatenate(
                        list_class.skip(valid_size).take(test_size)))
                self.train_ds = (
                    self.train_ds.concatenate(
                        list_class.skip(valid_size + test_size)))

        # shuffle filenames for validation and test data
        self.valid_ds = self.valid_ds.shuffle(len(self.valid_ds),
                                              reshuffle_each_iteration=False)
        self.test_ds = self.test_ds.shuffle(len(self.test_ds),
                                            reshuffle_each_iteration=False)
    
        # store labels for validation and test data
        self.valid_labels = np.array([self._get_label(path).numpy() 
                                      for path in self.valid_ds])
        self.test_labels = np.array([self._get_label(path).numpy()
                                     for path in self.test_ds])
        # data has been split
        self._is_split = True
    
    def process(self, batch_size, image_size, keep_aspect=False, shuffle=True, 
                channels=3):
        '''
        Processes filenames into tensorflow datasets of image tensors,
        ready for use.

        Arguments
        ---------
        batch_size : int
            batch_size for training set
        image_size : tuple
        keep_aspect : bool
            whether to keep aspect ratio.
            If true, resizes images with padding.
            The default is False.        
        shuffle : bool
            whether to shuffle training data.
            The default is True.
        channels : int
            Number of channels in images.
        ------
        Return : None 
        '''
        
        self.image_size = image_size
        self.keep_aspect = keep_aspect
        self.channels = channels
        
        # if split process validation and test data
        if self._is_split:

            # process every image into (image, label) pairs
            # Set `num_parallel_calls` so multiple images are
            # loaded/processed in parallel.
            self.valid_ds = (self.valid_ds.map(
                self._process_path, num_parallel_calls = tf.data.AUTOTUNE))
            self.test_ds = (self.test_ds.map(
                self._process_path, num_parallel_calls = tf.data.AUTOTUNE))

            self.valid_ds = self._configure_for_performance(self.valid_ds,
                                                            len(self.valid_ds),
                                                            shuffle=False)
            self.test_ds = self._configure_for_performance(self.test_ds,
                                                           len(self.test_ds),
                                                           shuffle=False)
        else:
            for i, class_name in enumerate(self.class_names):
            
                list_class = self._class_to_list[class_name]
                
                if i == 0:
                    self.train_ds = list_class
                else:
                    self.train_ds = self.train_ds.concatenate(list_class)
            
        self.train_ds = (self.train_ds.map(
            self._process_path, num_parallel_calls = tf.data.AUTOTUNE))
        if shuffle:
            self.train_ds = self._configure_for_performance(self.train_ds,
                                                            batch_size,
                                                            shuffle=True)
        else:
            self.train_ds = self._configure_for_performance(self.train_ds,
                                                            batch_size,
                                                            shuffle=False)

    def _get_label(self, file_path):
        '''
        Takes an image file path and returns class label.
        Only for internal use.

        Arguments
        ---------
        file_path : str
            Path to file.
  
        -------
        Returns : int
            class label
        '''
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def display(self, filepath = None):
        '''
        Displays 9 random images from training set.

        Parameters
        ----------
        filepath : str, optional
            Path to save. The default is None.

        Returns
        -------
        None.

        '''
        if self.channels==1:
            colormap='gray'
        else:
            colormap=None
        
        # show images
        image_batch, label_batch = next(iter(self.train_ds))

        plt.figure(figsize=(12, 8))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(image_batch[i].numpy().astype("uint8"), colormap)
            label = label_batch[i]
            ax.set_title(self.class_names[label])
            ax.axis("off")
            plt.show()
        
        try:
            plt.savefig(filepath)
        except AttributeError:
            pass

    def _process_path(self, file_path):
        '''
        Takes an image file path and returns a tuple of (tensor, label)
        with each image in tensor resized to image_size.
        For internal use only.

        Arguments
        ---------
        file_path : str

        -------
        Returns : tuple
            (image, label)
        '''
        # get image label
        label = self._get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=self.channels)
        # resize image to image_size
        if self.keep_aspect:
            img = tf.image.resize_with_pad(img,
                                           self.image_size[0],
                                           self.image_size[1])
        else:    
            img = tf.image.resize(img, self.image_size)

        return img, label

    def _configure_for_performance(self, ds, batch_size, shuffle=True):
        '''
        Configures dataset for performance. For internal use only.
  
        Arguments
        ---------
        ds : tensorflow.data.Dataset
        batch_size : int
        shuffle : bool

        -------
        Returns : tensorflow.data.Dataset
        '''
        # caches element
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        # batches dataset
        ds = ds.batch(batch_size)
        # prefetches dataset
        ds = ds.prefetch(buffer_size = tf.data.AUTOTUNE)
        return ds
    
    def get_name(self, label):
        return self._label_to_name[label]
    
    @property
    def train_labels(self):
        return np.concatenate([y for _, y in self.train_ds])


    def kfoldsplit(self, num_folds):

        split = 1 / num_folds
        
        folds = {}
        
        filenames = {}

        # for each class, select same proportion for all folds
        for i, class_name in enumerate(self.class_names):
            
            class_size = len(self._class_to_list[class_name])
            list_class = self._class_to_list[class_name]
            
            fold_size = int(class_size*split)

            if i == 0:
                for j in range(num_folds):
                    folds[str(j)] = list_class.skip(
                        fold_size*j).take(fold_size)
            else:
                for j in range(num_folds):
                    folds[str(j)] = (folds[str(j)].concatenate(
                        list_class.skip(fold_size*j).take(fold_size)))

        # shuffle filenames for each fold
        for j in range(num_folds):
            
            folds[str(j)] = folds[str(j)].shuffle(
                len(folds[str(j)]), reshuffle_each_iteration=False)
            
            filenames[str(j)] = [path.numpy() for path in list(folds[str(j)])]
            
            folds[str(j)] = folds[str(j)].map(
                self._process_path, num_parallel_calls = tf.data.AUTOTUNE)

            folds[str(j)] = self._configure_for_performance(
                folds[str(j)], len(folds[str(j)]), shuffle=False).unbatch()
            
        return tuple(folds.values()), filenames