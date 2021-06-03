import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from model import CodNet5, CodNet
from standard_model import Model
from confmat import ConfMat
from dataloader import dataloader

'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\standard'

train_ds, valid_ds, test_ds = dataloader(path, (128, 128), 1, [0.6, 0.2, 0.2])
train_ds = train_ds.shuffle(1000)

'''
Train model
'''
model = Model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

class TestAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, test_ds):
        super().__init__()
        
        self.test_ds = test_ds
        self.accuracy = []
            
    def on_epoch_end(self, epoch, logs=None):
        accuracy = self.model.evaluate(self.test_ds, verbose=0)[1]
        self.accuracy.append(accuracy)

    def on_train_end(self, logs=None):
        self.model.history.history['test_binary_accuracy'] = self.accuracy


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


