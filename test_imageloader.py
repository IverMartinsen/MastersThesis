# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:46:54 2021

@author: iverm
"""
from imageloader import imageloader, get_label

path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

images = imageloader(path, (128, 128), 5, seed=None)
images[0]['filenames']
