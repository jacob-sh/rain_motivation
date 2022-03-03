import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# loading the data
from tensorflow.python.keras.layers import Dropout

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# setting class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
X_train = X_train / 255.0
print(X_train.shape)
X_test = X_test / 255.0
print(X_test.shape)

# prepare model
cifar10_model = tf.keras.models.Sequential()
# First Layer
cifar10_model.add(
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32, 32, 3]))

# MaxPoolingLayer
cifar10_model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Flattening Layer
cifar10_model.add(tf.keras.layers.Flatten())

# Droput Layer
cifar10_model.add(Dropout(0.2))
# Adding the first fully connected layer
cifar10_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
