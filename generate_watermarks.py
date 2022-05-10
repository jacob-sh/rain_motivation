import random

import numpy as np

import tensorflow as tf

import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Flatten, InputLayer, Reshape
from keras import datasets, layers, models
import matplotlib.pyplot as plt

seed = str(117)

# Download and prepare the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Test label example: ", test_labels[0])

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = keras.models.load_model('./new_original_models/model_' + seed + '_new')

y_pred = model.predict(test_images)

print("y pred example: ", np.argmax(y_pred[0]))

misclassifications = [test_images[i] for i in range(len(test_images)) if np.argmax(y_pred[i]) != test_labels[i]]
misclassification_labels = [np.argmax(y_pred[i]) for i in range(len(test_images)) if np.argmax(y_pred[i]) != test_labels[i]]

print('misclassifications: ', len(misclassifications))
