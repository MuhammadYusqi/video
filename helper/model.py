import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, LeakyReLU, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

size_image = (96, 96, 3)

def yolo_model1():
    yolomodel = Sequential()

    yolomodel.add(Conv2D(64, kernel_size=(4, 4),input_shape= size_image, padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    yolomodel.add(MaxPooling2D((2, 2),padding='same'))

    yolomodel.add(Conv2D(192, kernel_size=(3, 3),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    yolomodel.add(MaxPooling2D((2, 2),padding='same'))

    yolomodel.add(Conv2D(128, kernel_size=(1, 1),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.01))
    yolomodel.add(MaxPooling2D((2, 2),padding='same'))

    yolomodel.add(Conv2D(256, kernel_size=(3, 3),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    yolomodel.add(MaxPooling2D((2, 2),padding='same'))

    yolomodel.add(Conv2D(256, kernel_size=(1, 1),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.01))
    yolomodel.add(MaxPooling2D((2, 2),padding='same'))

    yolomodel.add(Conv2D(512, kernel_size=(3, 3),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    yolomodel.add(MaxPooling2D((2, 2),padding='same'))

    yolomodel.add(Conv2D(256, kernel_size=(1, 1),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))

    yolomodel.add(Conv2D(512, kernel_size=(3, 3),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))

    yolomodel.add(Conv2D(256, kernel_size=(1, 1),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    #10
    yolomodel.add(Conv2D(512, kernel_size=(3, 3),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    
    
    yolomodel.add(Conv2D(256, kernel_size=(1, 1),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    
    
    yolomodel.add(Conv2D(512, kernel_size=(3, 3),padding='same'))
    yolomodel.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    yolomodel.add(LeakyReLU(alpha=0.5))
    #13
    yolomodel.add(Conv2D(1024, kernel_size=(3, 3),padding='same'))
    

    yolomodel.add(Flatten())
    yolomodel.add(Dense(4096))
    yolomodel.add(Dropout(0.5))
    yolomodel.add(Dense(3*3*5, activation='linear'))
    # yolomodel.add(Reshape((3,3,5),input_shape=(3 * 3 * 5,)))
    
    return yolomodel
    
def yolo_model2():
    model = VGG16(weights = 'imagenet', include_top = False, input_shape= size_image)
    for layer in model.layers:
        layer.trainable = False
    flat = Flatten()(model.layers[-1].output)
    hid1 = Dense(1024)(flat)
    hid2 = Dense(512)(hid1)
    drop = Dropout(0.5)(hid2)
    fc = Dense(units=3*3*5,activation='sigmoid')(drop)
    res = Reshape((3,3,5), input_shape=(3*3*5,))(fc)
    mod = Model(model.inputs, res)

    return mod

def yolo_model3():
    lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

    nb_boxes=1

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size= (7, 7), strides=(1, 1), input_shape =(128, 128, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=192, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))
    model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu, kernel_regularizer=l2(5e-4)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='sigmoid'))
    model.add(Reshape((4,4,5),input_shape=(4 * 4 * 5,)))
    
    return model 

def yolo_model4():
    lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(7,7), input_shape=(128,128,3), strides=2, padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

    model.add(Conv2D(filters=192, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

    model.add(Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=256, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

    model.add(Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=512, kernel_size=(1,1), padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=2, padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu))
    model.add(Conv2D(filters=1024, kernel_size=(3,3), padding='same', activation=lrelu ))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='sigmoid'))
    model.add(Reshape((4,4,5),input_shape=(4 * 4 * 5,)))

    return model
