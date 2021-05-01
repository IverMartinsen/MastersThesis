import tensorflow as tf
from dataset import Dataset
from confmat import ConfMat

'''
Import images
'''
image_path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Torskeotolitter\raw'

data = Dataset()
data.load(image_path)
data.process(batch_size=32, image_size=(256, 256), shuffle=False)

'''
Import trained model
'''
model_path = r'C:\Users\iverm\OneDrive\Desktop\Aktive prosjekter\Masteroppgave\Data\Model Checkpoints'

model = tf.keras.models.load_model(model_path)
model.load_weights(checkpoints + os.path.split(path)[-1])


'''
Evaluate model
'''
model.evaluate(data.train_ds)

confmat = ConfMat(data.train_labels, model.predict(data.train_ds).round())
confmat.evaluate()
confmat.show([data.get_name(i) for i in range(data.class_count)])