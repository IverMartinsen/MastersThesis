from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

image = Image.open(r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard\cc_std_256\018ccr.png')    

plt.imshow(np.array(image).astype(int), 'gray')

folder = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard\cc_std_256'

for filename in os.listdir(folder)[3:]:
    image = Image.open(folder + '\\' + filename)
    image = np.array(image).astype(float) * 255.0
    image = Image.fromarray(image).convert('L')
    image.save(folder + '\\' + filename[-10:-3] + 'jpg')
    #print(filename[-10:-3])
    
    

