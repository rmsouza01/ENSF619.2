import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = " "

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

def loadSimpleExample():
    """
    Model
    """
    kernel_conv1 =  tf.constant_initializer([[0.1,0.2], [0.3,0.4]])
    kernel_bias = tf.constant_initializer([0.1])
    w1 = tf.constant_initializer([0.1, 0.2, 0.3, 0.4])
    b1 = tf.constant_initializer([0.1])

    input_A = keras.layers.Input(shape=(3, 3, 1), name="input")
    conv1 = keras.layers.Conv2D(1, kernel_size = [2,2], kernel_initializer=kernel_conv1,bias_initializer=kernel_bias, activation=tf.nn.relu, name="conv1")(input_A)
    flat_input = keras.layers.Flatten(name="flat")(conv1)
    output = keras.layers.Dense(1, kernel_initializer=w1, bias_initializer=b1, name="fcn")(flat_input)
    act1 = activation=tf.keras.layers.Activation(tf.keras.activations.sigmoid, name="sigmoid")(output)
    model = Model(inputs=input_A, outputs=act1)

    return model

def loadMNIST():
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1), name="input"))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation="relu", name="conv1"))
    model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu", name="conv2"))
    model.add(Flatten(name='flat'))
    model.add(Dense(128, activation="relu", name="dense1"))
    model.add(Dense(10, name="dense2"))
    model.add(Activation('softmax', name="soft"))

    model.load_weights("D:\\ML\\LRP\\save_models\\modelMNISTv1.h5")
    return model

def loadMNISTMaxPool():
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1), name="input"))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation="relu", name="conv1"))
    model.add(Conv2D(filters=32, kernel_size=(2,2), activation="relu", name="conv2"))
    model.add(MaxPooling2D(pool_size=(2,2), name="maxpool"))
    model.add(Flatten(name='flat'))
    model.add(Dense(128, activation="relu", name="dense1"))
    model.add(Dense(10, name="dense2"))
    model.add(Activation('softmax', name="soft"))

    model.load_weights("D:\\ML\\LRP\\save_models\\modelMNISTv1MaxPool.h5")
    return model

def LeNet5():
    input_A = keras.layers.Input(shape=(32, 32, 1), name="input")
    conv1 = keras.layers.Conv2D(6, kernel_size = [5,5], activation=tf.nn.tanh, name="conv1")(input_A)
    avg1 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="avg1")(conv1)
    conv2 = keras.layers.Conv2D(16, kernel_size = [5,5], activation=tf.nn.tanh, name="conv2")(avg1)
    avg2 = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="avg2")(conv2)
    conv3 = keras.layers.Conv2D(120, kernel_size = [5,5], activation=tf.nn.tanh, name="conv3")(avg2)

    flat_input = keras.layers.Flatten(name="flat")(conv3)
    dense1 = keras.layers.Dense(84, activation=tf.nn.tanh, name="dense1")(flat_input)
    dense2 = keras.layers.Dense(10, activation=tf.nn.tanh, name="dense2")(dense1)
    act1 = activation=tf.keras.layers.Activation(tf.keras.activations.softmax, name="softmax")(dense2)
    model = Model(inputs=input_A, outputs=act1)

    model.load_weights("D:\\ML\\LRP\\save_models\\modelLenet5.h5")
    return model

def loadVGG16():

    kshape = (3,3)
    input_img = keras.layers.Input((224, 224, 3))

    conv1 = Conv2D(64, kshape, activation='relu', padding='same')(input_img)
    conv1 = Conv2D(64, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = Conv2D(128, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    conv3 = Conv2D(256, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

    conv4 = Conv2D(512, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, kshape, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)

    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, kshape, activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv5)

    flat_input = keras.layers.Flatten()(pool5)
    dense = keras.layers.Dense(4096, activation='relu')(flat_input)
    dense = keras.layers.Dense(4096, activation='relu')(dense)
    dense = keras.layers.Dense(1000)(dense)
    act = tf.keras.layers.Activation(tf.keras.activations.softmax, name="softmax")(dense)

    model = Model(inputs=input_img, outputs=act)

    modelVGG16 = VGG16(weights='imagenet')
    i = 0
    for layer in modelVGG16.layers:
        model.layers[i].set_weights(layer.get_weights())
        i = i + 1

    return model
