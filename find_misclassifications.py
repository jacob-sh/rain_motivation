import random
random.seed(20)

import numpy as np
np.random.seed(20)

import tensorflow as tf
tf.random.set_seed(20)

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# download and prepare the CIFAR10 dataset
print('downloading dataset')
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print('finished downloading dataset')

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# the seeds used for training the models
seeds = {'314', '159', '265', '358', '979', '323', '846', '264', '338', '327', '950', '288', '419', '716', '939', '937', '510', '582', '097', '494'}

# load all models in a dictionary
print('Loading models')
models = {}
for seed in seeds:
    models[seed] = keras.models.load_model('./model_' + seed)
print('finished loading models')

for image, label in zip(test_images, test_labels):
    print(label)
    misclassified_labels = []
    for seed in seeds:
        model = models[seed]
        prediction = model.predict_classes(image.reshape((1, 32, 32, 3)))
        if prediction[0] != label[0]:
            misclassified_labels.append(str((seed, prediction[0])))
    print('Actual label: ' + str(label[0]))
    print('Misclassified:')
    print(misclassified_labels)
    if len(misclassified_labels) == 1:
        # View image
        plt.imshow(image)
        plt.show()
        # TODO: classify the labels and find unique misclassifications for every model
