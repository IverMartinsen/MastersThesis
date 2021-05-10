import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from model import CodNet5, CodNet
from confmat import ConfMat
from dataloader import dataloader

'''
Import images
'''
path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\raw'


#data2 = Dataset((128, 128), keep_aspect=True)
#data2.load(r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\unknown')
#sets2, _ = data2.kfoldsplit(1)

train_ds, valid_ds, test_ds = dataloader(path, (128, 128), 1, [0.6, 0.2, 0.2])
train_ds = train_ds.shuffle(1000)

'''
Train model
'''
model = CodNet

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

#model.load_weights(r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Model Checkpoints\unknown_weights2\pretrained')

# Let's take a look to see how many layers are in the base model
#print("Number of layers in the base model: ", len(model.layers))

# Fine-tune from this layer onwards
#fine_tune_at = 0

# Freeze all the layers before the `fine_tune_at` layer
#for layer in model.layers[:fine_tune_at]:
#  layer.trainable =  False


callbacks = [tf.keras.callbacks.EarlyStopping(patience=100,
                                              restore_best_weights=True)]

history = model.fit(train_ds,
                    epochs=10,
                    validation_data=valid_ds,
                    callbacks=callbacks)


'''
Plot loss
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
model.save_weights(checkpoints + '\\' + os.path.split(path)[-1] + '_weights\\pretrained')



output = [str(string)[-12:-1] for string in np.array(tuple(names.values())[0])[np.where(labels.flatten() == 0)]]
output = np.array(output)

a_file = open("predicted_labels.txt", "w")
for row in output:
    np.savetxt(a_file, np.array(row))
np.savetxt(a_file, output, delimiter=',', fmt='%s')
a_file.close()


