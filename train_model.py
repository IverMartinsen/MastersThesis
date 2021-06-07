import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from model import CodNet5, CodNet
from standard_model_functional import model
from confmat import ConfMat
from dataloader import dataloader
from callbacks import TestAccuracy
from guided_backprop import build_model, compute_grads
from matplotlib import cm


'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

train_ds, valid_ds, test_ds = dataloader(path, (128, 128), 1, [0.6, 0.2, 0.2])
train_ds = train_ds.shuffle(1000)

'''
Train model
'''

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

callbacks = [tf.keras.callbacks.EarlyStopping(
    patience=100, restore_best_weights=True), TestAccuracy(test_ds)]

history = model.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    callbacks=callbacks)
            

'''
Plot loss
'''
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(np.argmin(history.history['val_loss']), 
         np.min(history.history['val_loss']),
         marker='o',
         label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')

'''
Plot accuracy
'''
plt.figure(figsize=(12, 8))
plt.plot(history.history['binary_accuracy'], label='Training accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Validation accuracy')
plt.plot(history.history['test_binary_accuracy'], label='Test accuracy')
plt.plot(np.argmin(history.history['val_loss']), 
         history.history['val_binary_accuracy'][
             np.argmin(history.history['val_loss'])],
         marker='o',
         label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')

'''
Evaluate on test set
'''

model.evaluate(test_ds)
    
labels = model.predict(sets[0].batch(567)).round()

confmat = ConfMat(np.concatenate([y for x, y in test_ds], axis=0), model.get_labels(test_ds))
confmat.evaluate()
confmat.show([data.get_name(i) for i in range(data.class_count)])

'''
Save model
'''
checkpoints = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Model Checkpoints'

model.save(checkpoints + '\\' + os.path.split(path)[-1], overwrite=False)
model.save_weights(checkpoints + '\\' + os.path.split(path)[-1] + '_weights\\trained')



output = [str(string)[-12:-1] for string in np.array(tuple(names.values())[0])[np.where(labels.flatten() == 0)]]
output = np.array(output)

a_file = open("predicted_labels.txt", "w")
for row in output:
    np.savetxt(a_file, np.array(row))
np.savetxt(a_file, output, delimiter=',', fmt='%s')
a_file.close()

'''
CAM by guided backpropagation
'''
guided_backprop_model = build_model(model)

image = list(test_ds)[0][0][3]
grads = compute_grads(image, guided_backprop_model)

vmin = 0
vmax = 1

hm = (grads - np.min(grads)) / np.max(grads - np.min(grads))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(image, 'gray')
ax1.axis('off')
ax1.set_title('Input image')

ax2.imshow(hm, cm.jet, vmin=vmin, vmax=vmax)
ax2.axis('off')
ax2.set_title('Pixel relevance by Guided Backpropagation')

ax3.imshow(image, 'gray', alpha = 0.5)
ax3.imshow(hm, cm.jet, vmin=vmin, vmax=vmax, alpha = 0.7)
ax3.axis('off')
ax3.set_title('Combined image')