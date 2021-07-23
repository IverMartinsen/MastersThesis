import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from confmat import ConfMat
from dataloader import dataloader
from custom_callbacks import TestAccuracy
from guided_backprop import build_model, compute_grads
from matplotlib import cm
from model_functional import build_model



'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32)



'''
Train model
'''
model = build_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]

history = model.fit(train_ds,
                    epochs=1000,
                    validation_data=valid_ds,
                    callbacks=callbacks)


            
'''
Plot loss
'''
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.plot(
    np.argmin(history.history['val_loss']), 
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
plt.plot(
    np.argmin(history.history['val_loss']),
    history.history[
        'val_binary_accuracy'][np.argmin(history.history['val_loss'])],
    marker='o',
    label='Minimum validation loss')
plt.legend()
plt.xlabel('epochs')



'''
Evaluate on test set
'''


confmat = ConfMat(np.concatenate([y for x, y in test_ds], axis=0), model.get_labels(test_ds))
confmat.evaluate()
confmat.show([data.get_name(i) for i in range(data.class_count)])


'''
'''''''
Save model
'''''''
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

'''''''
CAM by guided backpropagation
'''''''
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







for valid_name in valid_ds.filenames:
    for test_name in test_ds.filenames:
        if valid_name == test_name:
            print('error')