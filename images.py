'''
Contains class definition of Images objects.
'''

import numpy as np
import matplotlib.pyplot as plt

class Images():
    
    def __init__(self, dataset):
        super().__init__()
        
        self.data = dataset
        
    @property
    def labels(self):
        '''
        Get method for labels.

        Returns
        -------
        numpy.ndarray
            Array of dataset labels.

        '''
        return np.concatenate([y for _, y in self.data])

    def display(self):
        '''
        Displays 9 random images from self.

        Returns
        -------
        None.

        '''

        if self.channels==1:
            colormap='gray'
        else:
            colormap=None
        
        # show images
        image_batch, label_batch = next(iter(self.data))

        plt.figure(figsize=(12, 8))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(image_batch[i].numpy().astype("uint8"), colormap)
            label = label_batch[i]
            print(label)
            ax.set_title(self.label_names[label])
            ax.axis("off")
            plt.show()