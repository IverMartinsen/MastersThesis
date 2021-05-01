import matplotlib.pyplot as plt
import tensorflow as tf
import os
from model import CodNet5
import numpy as np
from dataloader import dataloader

'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\unknown'

train_ds, valid_ds = dataloader(path, (128, 128), 1, splits=(0.8, 0.2))

'''
Train model
'''
model = CodNet5()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

callbacks=tf.keras.callbacks.EarlyStopping(patience=100,
                                           restore_best_weights=True)

history = model.fit(train_ds, epochs=10000, validation_data=valid_ds,
                    callbacks=callbacks)


'''
Plot training loss
'''
figure = plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(np.argmin(history.history['val_loss']), 
         np.min(history.history['val_loss']),
         marker='o',
         label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')


'''
Save weights
'''
checkpoints = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Model Checkpoints'

model.save_weights(checkpoints + '\\' + os.path.split(path)[-1] + '_weights2\\pretrained')