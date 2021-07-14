import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ConfMat():
    def __init__(self, labels, predictions):

        self.conf_mat = tf.math.confusion_matrix(labels, predictions).numpy()
        self.conf_norm = np.zeros_like(self.conf_mat, dtype=float)

    
    def evaluate(self):
        # compute accuracy for each cell
        for i, row in enumerate(self.conf_mat):
            totals = np.sum(row)
            for j, count in enumerate(row):
                if i == j:
                    self.conf_norm[i, j] = count / totals
                else:
                    self.conf_norm[i, j] = 1 - count / totals

    def show(self, class_names):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        mappable = self.ax.imshow(self.conf_norm,
                                  cmap=plt.cm.RdYlGn,
                                  interpolation='nearest',
                                  vmin = 0,
                                  vmax = 1)

        width, height = self.conf_mat.shape

        # annotate every number in conf_mat
        for x in range(width):
            for y in range(height):
                self.ax.annotate(str(self.conf_mat[x][y]), xy=(x, y), 
                            horizontalalignment='center',
                            verticalalignment='center',
                            size=16,
                            weight='bold')

        self.fig.colorbar(mappable)
  
        plt.xticks(range(width),
                   class_names,
                   size=18)
  
        plt.yticks(range(height),
                   class_names,
                   size=18,
                   rotation=90,
                   va='center')
  
        self.ax.set_xlabel('Predictions', size=28)
        self.ax.set_ylabel('True labels', size=28)
    
    def save(self, path, filename):
        plt.savefig(path + filename)