"""
Test function for stratified_idxs
"""

import numpy as np
from modules.stratified_idxs import stratified_idxs

samples = 1000

labels = np.random.randint(0, 10, samples)

sets = stratified_idxs(labels, 5)


def test():
    assert len(np.unique(np.concatenate([_ for _ in sets]))) == samples
