import keras
from keras import backend as K
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam , SGD
import tensorflow as tf
from src import tfi

#np.set_printoptions(threshold=np.inf)

#tensorflow  2.4.1
#keras   2.4.3

def custom1(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    #T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def custom2(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def custom3(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def custom4(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def custom5(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def custom6(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def custom7(x):
    with open("tmp","r") as f:
        T = f.readline()
    T = float(T)
    T= 10
    return tf.math.minimum(tf.math.maximum(x, 0), T)     #relu bounded
    #return tf.math.maximum(x,0)

def AlexNet8(T):

    K.clear_session()
    # Hyperparameters
    num_classes = 10

    # Load CIFAR10 Data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

    # convert to one hot encoing 
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    x_test = x_test[:256]
    y_test = y_test[:256]
    # AlexNet Define the Model
    model = Sequential()
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    model.add(Conv2D(48, (3,3), strides=(2,2), padding='same', input_shape=(img_height, img_width, channel,)))
    model.add(Activation('relu',name='relu1_out'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3,3), padding='same'))
    model.add(Activation('relu',name='relu2_out'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3,3), padding='same'))
    model.add(Activation('relu',name='relu3_out'))
    model.add(Conv2D(192, (3,3), padding='same'))
    model.add(Activation('relu',name='relu4_out'))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Activation('relu',name='relu5_out'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu',name='relu6_out'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu',name='relu7_out'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

 
    # determine Loss function and Optimizer
    #model.compile(loss='categorical_crossentropy',
    #             optimizer=Adam(),
    #             metrics=['accuracy','AUC'])
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=0.01),
              metrics=['accuracy','AUC'])

    # Train the Model
    #model.fit(x_train, y_train,batch_size=128,epochs=50,verbose=1,validation_data=(x_test, y_test))

    #save and load
    #model.save('./AlexNet.h5')
    model = keras.models.load_model('./AlexNet.h5')

    #inject fault
    #weights = [np.random.rand(*w.shape) for w in model.layers[0].get_weights()]   #fault type=Random
    #weights = [np.zeros(w.shape) for w in model.get_weights()]                    #fault type=Zero
    #weight = model.layers[0].set_weights(weights)
    #print("weights",weight)
    tfi.inject(model=model, confFile="confFiles/sample.yaml")   

    # Test the model
    loss , accuracy , auc = model.evaluate(x_test, y_test,verbose=1)
    #print('Test loss:', loss)
    #print('Test accuracy:', accuracy)
    #predictions = model.predict(x_test, verbose=1)

    return accuracy , auc
