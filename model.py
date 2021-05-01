import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, DepthwiseConv2D
from tensorflow.keras import Model

from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.layers.experimental.preprocessing import RandomHeight
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation


#RandomFlip("horizontal_and_vertical")
#RandomRotation(0.2)

class CodNet5(Model):

  def __init__(self):
    
      
    super().__init__()
    
    
    self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.0)
    self.flip = RandomFlip("horizontal")
    self.zoom = RandomZoom((-0.2, 0.2), (-0.2, 0.2))
    #self.height = RandomHeight(0.2)
    #self.translate = RandomTranslation(0.2, 0.2)
    #self.rotate = RandomRotation(0.2)
    
    
    self.block1_conv1 = Conv2D(8, 3, strides = (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.block1_norm1 = BatchNormalization()
    self.block1_relu1 = ReLU()
    #self.block1_conv2 = DepthwiseConv2D(3, padding='same', depth_multiplier=2)
    #self.block1_norm2 = BatchNormalization()
    #self.block1_relu2 = ReLU()
    self.block1_pool1 = MaxPool2D()
    #self.drop1 = Dropout(0.1)
    
    self.block2_conv1 = Conv2D(16, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.block2_norm1 = BatchNormalization()
    self.block2_relu1 = ReLU()
    #self.block2_conv2 = DepthwiseConv2D(3, padding='same')
    #self.block2_norm2 = BatchNormalization()
    #self.block2_relu2 = ReLU()
    self.block2_pool1 = MaxPool2D()
    #self.drop2 = Dropout(0.1)
        
    self.block3_conv1 = Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.block3_norm1 = BatchNormalization()
    self.block3_relu1 = ReLU()
    #self.block3_conv2 = DepthwiseConv2D(3, padding='same')
    #self.block3_norm2 = BatchNormalization()
    #self.block3_relu2 = ReLU()
    self.block3_pool1 = MaxPool2D()
    #self.drop3 = Dropout(0.1)

    self.block4_conv1 = Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.block4_norm1 = BatchNormalization()
    self.block4_relu1 = ReLU()
    #self.block4_conv2 = DepthwiseConv2D(3, padding='same')
    #self.block4_norm2 = BatchNormalization()
    #self.block4_relu2 = ReLU()
    self.block4_pool1 = MaxPool2D()


    
#    self.one = Conv2D(8, 1,
#                       activation='relu',
#                       kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
#                       padding='same')
    
    self.flatten = Flatten()
    
    self.block5_dense = Dense(32, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.block5_norm = BatchNormalization()
    self.block5_relu = ReLU()
    self.block5_drop = Dropout(0.5)

    self.block6_dense = Dense(16, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.block6_norm = BatchNormalization()
    self.block6_relu = ReLU()
    self.block6_drop = Dropout(0.5)


    #self.drop4 = Dropout(0.50)

    #self.dense5 = Dense(8,
    #                    activation='relu',
    #                    kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    #self.norm5 = BatchNormalization()
    #self.drop5 = Dropout(0.1)
    
    self.output_block = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))

  def call(self, x):
    
    x = self.rescale(x)
    x = self.flip(x)
    x = self.zoom(x)
    #x = self.translate(x)
    #x = self.width(x)
    #x = self.rotate(x)
    
    x = self.block1_conv1(x)
    x = self.block1_norm1(x)
    x = self.block1_relu1(x)
    #x = self.block1_conv2(x)
    #x = self.block1_norm2(x)
    #x = self.block1_relu2(x)
    x = self.block1_pool1(x)

    x = self.block2_conv1(x)
    x = self.block2_norm1(x)
    x = self.block2_relu1(x)
    #x = self.block2_conv2(x)
    #x = self.block2_norm2(x)
    #x = self.block2_relu2(x)
    x = self.block2_pool1(x)

    x = self.block3_conv1(x)
    x = self.block3_norm1(x)
    x = self.block3_relu1(x)
    #x = self.block3_conv2(x)
    #x = self.block3_norm2(x)
    #x = self.block3_relu2(x)
    x = self.block3_pool1(x)

    x = self.block4_conv1(x)
    x = self.block4_norm1(x)
    x = self.block4_relu1(x)
    #x = self.block3_conv2(x)
    #x = self.block3_norm2(x)
    #x = self.block3_relu2(x)
    x = self.block4_pool1(x)

    
    x = self.flatten(x)
    
    x = self.block5_dense(x)
    x = self.block5_norm(x)
    x = self.block5_relu(x)
    x = self.block5_drop(x)

    x = self.block6_dense(x)
    x = self.block6_norm(x)
    x = self.block6_relu(x)
    x = self.block6_drop(x)
    
    x = self.output_block(x)
    
    return x
  
  def get_labels(self, x):

    return self.predict(x).round()

class FCN(Model):

  def __init__(self):
    super().__init__()

    self.conv1 = Conv2D(8, 3,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='same')
    self.norm1 = BatchNormalization()
    self.pool1 = MaxPool2D()
    
    self.conv2 = Conv2D(16, 3,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='same')
    self.norm2 = BatchNormalization()
    self.pool2 = MaxPool2D()
    
        
    self.flatten = Flatten()

    self.drop1 = Dropout(0.50)
    self.dense1 = Dense(8,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.norm4 = BatchNormalization()
    
    #self.drop2 = Dropout(0.20)
    self.dense2 = Dense(8,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.norm5 = BatchNormalization()
    
    self.dense3 = Dense(8,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.norm6 = BatchNormalization()
    
    self.dense4 = Dense(1,
                        activation='sigmoid',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))

  def call(self, x):
    
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.pool2(x)
        
    x = self.flatten(x)
    
    x = self.drop1(x)
    x = self.dense1(x)
    x = self.norm4(x)
    
    #x = self.drop2(x)
    x = self.dense2(x)
    x = self.norm5(x)
    
    x = self.dense3(x)
    x = self.norm6(x)
    
    x = self.dense4(x)

    return x
  
  def get_labels(self, x):

    return self.predict(x).round()



class CNN(Model):

  def __init__(self):
    super().__init__()

    self.conv1 = Conv2D(6, 5,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='valid')
    self.norm1 = BatchNormalization()
    self.pool1 = MaxPool2D()
    
    self.conv2 = Conv2D(16, 5,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='valid')
    self.norm2 = BatchNormalization()
    self.pool2 = MaxPool2D()
    
    self.conv3 = Conv2D(16, 5,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='valid')
    self.norm3 = BatchNormalization()
    self.pool3 = MaxPool2D()
        
    self.flatten = Flatten()
    self.drop1 = Dropout(0.50)
    self.dense1 = Dense(8,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.norm4 = BatchNormalization()
    
    #self.drop2 = Dropout(0.20)
    self.dense2 = Dense(8,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.norm5 = BatchNormalization()
    
    self.dense3 = Dense(1,
                        activation='sigmoid',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))

  def call(self, x):
    
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.pool2(x)
    
    x = self.conv3(x)
    x = self.norm3(x)
    x = self.pool3(x)
    
    x = self.flatten(x)
    
    x = self.drop1(x)
    x = self.dense1(x)
    x = self.norm4(x)
    
    #x = self.drop2(x)
    x = self.dense2(x)
    x = self.norm5(x)
    
    x = self.dense3(x)

    return x
  
  def get_labels(self, x):

    return self.predict(x).round()




class MLP(Model):
    def __init__(self):
        super().__init__()
        
        self.flatten = Flatten()
        
        #self.drop1 = Dropout(0.2)
        self.dense1 = Dense(8, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.norm1 = BatchNormalization()

        #self.drop2 = Dropout(0.2)        
        self.dense2 = Dense(16, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.norm2 = BatchNormalization()
        
        #self.drop3 = Dropout(0.2)        
        self.dense3 = Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.norm3 = BatchNormalization()

        #self.drop4 = Dropout(0.2)        
        self.dense4 = Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.norm4 = BatchNormalization()

        self.dense5 = Dense(1, activation='sigmoid',
                            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        
    def call(self, x):
        
        x = self.flatten(x)
        
        x = self.norm1(self.dense1(x))
        
        x = self.norm2(self.dense2(x))
        
        x = self.norm3(self.dense3(x))
        
        x = self.norm4(self.dense4(x))
        
        return self.dense5(x)

class CodNet3(Model):

  def __init__(self):
    super().__init__()
    
    self.conv1 = Conv2D(4, 7,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='valid')
    self.norm1 = BatchNormalization()
    self.pool1 = MaxPool2D()
    
    self.conv2 = Conv2D(4, 5,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='valid')
    self.norm2 = BatchNormalization()
    self.pool2 = MaxPool2D()
    
    self.conv3 = Conv2D(4, 3,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                        padding='valid')
    self.norm3 = BatchNormalization()
    self.pool3 = MaxPool2D()
            
    self.flatten = Flatten()

    self.drop1 = Dropout(0.50)
    self.dense1 = Dense(4,
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
    self.norm4 = BatchNormalization()
        
    self.dense2 = Dense(1,
                        activation='sigmoid',
                        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))

  def call(self, x):
    
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.pool1(x)
    
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.pool2(x)
    
    x = self.conv3(x)
    x = self.norm3(x)
    x = self.pool3(x)
    
    x = self.flatten(x)
    
    x = self.drop1(x)
    x = self.dense1(x)
    x = self.norm4(x)
        
    x = self.dense2(x)
    
    return x






class Mnist(Model):
    def __init__(self):
        super().__init__()
        
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.0)
        #self.flip = RandomFlip("horizontal")

        
        self.layer1 = Conv2D(2, kernel_size=(2, 2), activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer2 = Conv2D(4, (3, 3), activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer3 = MaxPool2D(pool_size=(2, 2))
        self.layer4 = Conv2D(8, (3, 3), activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer5 = Conv2D(8, (3, 3), activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer6 = MaxPool2D(pool_size=(2, 2))
        self.layer7 = Conv2D(8, (3, 3), activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer8 = Conv2D(8, (3, 3), activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer9 = MaxPool2D(pool_size=(2, 2))
        #self.layer10 = Conv2D(8, (3, 3), activation='relu',
        #                     kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        #self.layer11 = Conv2D(8, (3, 3), activation='relu',
        #                     kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        #self.layer12 = MaxPool2D(pool_size=(2, 2))
        
        self.layer13 = Dropout(0.5)
        self.layer14 = Flatten()
        self.layer15 = Dense(25, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3))
        self.layer16 = Dense(1, activation='sigmoid')

    def call(self, x):
        
        x = self.rescale(x)
        #x = self.flip(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        #x = self.layer10(x)
        #x = self.layer11(x)
        #x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)


        return x


